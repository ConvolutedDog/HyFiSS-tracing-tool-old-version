#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <map>
#include <list>
#include <bitset>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <sstream>

#include "utils/utils.h"
#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/channel.hpp"
#include "common.h"

using namespace std;

#define MAX_KERNELS 300

/* channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
* flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
 pthread_mutex_t mutex;

/* opcode to id map and reverse map  */
map<string, int> opcode_to_id_map;
map<int, string> id_to_opcode_map;


typedef unordered_map<int, int> int_int_map;

// int_int_map bb_map;
int kernel_id = 1;
int bb_id = 0;
int current_sm_id;
int current_cta_id_x;
int current_cta_id_y;
int current_cta_id_z;
int current_warp_id;
int first_warp_exec = 0;
int first_kernel_mem_clk = 0;
static ofstream app_config_fp;
static ofstream insts_trace_fp;
static ofstream issue_config_fp; // yangjianchao16 add
static ofstream instn_config_fp; // yangjianchao16 add
int_int_map reg_dependency_map;
int_int_map pred_dependency_map;
int inst_count = 0;

int kernel_gridX = 0;
int kernel_gridY = 0;
int kernel_gridZ = 0;


typedef map<int, vector<tuple<int, int, int, int>>> SMid_CTAid_Map_t;

/* This map intends to record the SM id that every CTA is issued to during 
 * the exection of one kernel. The key and value of this map:
 * key: CTA id, value: <kernel_id, ctaid.x, ctaid.y, ctaid.z>
 */
SMid_CTAid_Map_t SMid_CTAid_Map;

typedef map<int, vector<uint64_t>> SMid_CTAid_timestamp_Map_t;

SMid_CTAid_timestamp_Map_t SMid_CTAid_timestamp_Map;


// typedef map<int, list<tuple<int, int, int, int, int>>> PCid_CTAid_Map_t;

/* This map intends to record the pc that every instn actually executed on 
 * the fly belongs to during the exection of one kernel. The key and value 
 * of this map:
 * key: <kernel_id, pc, ctaid.x, ctaid.y, ctaid.z, warp_id>, value: string
 */
// PCid_CTAid_Map_t PCid_CTAid_Map;


typedef map<tuple<int, int>, string> PCid_Instn_Map_t;
/* This map intends to record the instruction string that actually executed 
 * on the fly belongs to during the exection of one kernel. The key and value 
 * of this map:
 * key: <kernel_id, pc>, value: string
 */
PCid_Instn_Map_t PCid_Instn_Map;

map<int, int> max_pc;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    string pad(100, '-');
    printf("%s\n", pad.c_str());
    
    if (mkdir("configs", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1){
        if( errno == EEXIST ) {
            // alredy exists
            system("rm configs/*");
        } else {
            // something else
            cout << "cannot create configs directory error:" << strerror(errno) << endl;
            throw runtime_error(strerror(errno));
            return;
        }
    }

    if (mkdir("memory_traces", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1){
        if( errno == EEXIST ) {
        // alredy exists
        system("rm memory_traces/*");
        } else {
        // something else
            cout << "cannot create memory_traces directory error:" << strerror(errno) << endl;
            throw runtime_error(strerror(errno));
            return;
        }
    }

    if (mkdir("sass_traces", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1){
        if( errno == EEXIST ) {
        // alredy exists
        system("rm sass_traces/*");
        } else {
        // something else
            cout << "cannot create sass_traces directory error:" << strerror(errno) << endl;
            throw runtime_error(strerror(errno));
            return;
        }
    }

    app_config_fp.open("./configs/app.config");
    app_config_fp << "####################################################################################\n";
    app_config_fp << "######                                                                        ######\n";
    app_config_fp << "######                 The app.config file for the simulator.                 ######\n";
    app_config_fp << "######                                                                        ######\n";
    app_config_fp << "####################################################################################\n\n";

    issue_config_fp.open("./configs/issue.config");
    issue_config_fp << "####################################################################################\n";
    issue_config_fp << "######                                                                        ######\n";
    issue_config_fp << "######                The issue.config file for the simulator.                ######\n";
    issue_config_fp << "######                                                                        ######\n";
    issue_config_fp << "####################################################################################\n";
    issue_config_fp << "\n";
    issue_config_fp << "# trace_issued_sms_num : the number of SMs that have issued at least one warp.\n";
    issue_config_fp << "# trace_issued_sm_id_x ctas_num : the number of CTAs that have been issued on SM x.\n";
    issue_config_fp << "# trace_issued_sm_id_x tuple_list : the list of CTAs that have been issued on SM x,\n";
    issue_config_fp << "#                                   format (kernel_id, cta.x, cta.y, cta.z)\n";
    issue_config_fp << "# trace_issued_sm_id_x is the list of issued CTAs on SM x.\n";
    issue_config_fp << "# Note that the list of issued CTAs on SM x, does not represent the order in which\n";
    issue_config_fp << "# CTAs are issued.\n\n";

    instn_config_fp.open("./configs/instn.config");
    instn_config_fp << "####################################################################################\n";
    instn_config_fp << "######                                                                        ######\n";
    instn_config_fp << "######                The instn.config file for the simulator.                ######\n";
    instn_config_fp << "######                                                                        ######\n";
    instn_config_fp << "####################################################################################\n\n";
}


void dump_app_config(){
    app_config_fp << "-app_kernels_id ";
    for (int i = 1; i < kernel_id; i++){
        if (i > 1) {
            app_config_fp << ",";
        }
        app_config_fp << i;
    }
    app_config_fp.close();
    cout << "--> sass + memory traces are collected for "<<(kernel_id - 1) << " kernels"<<"\n";
}

/* set used to avoid re-instrumenting the same functions multiple times */
unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* get related functions of the kernel (device function that can be
     * called by the kernel) */
    vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);
         

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        // /* get the static control flow graph of instruction */
        // const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        // if (cfg.is_degenerate) {
        //     printf("Warning: Function %s is degenerated, we can't compute basic "
        //         "blocks statically",
        //         nvbit_get_func_name(ctx, f));
        // }
        // int bb_id = 0;
        // int count = 0;
        // cout<<"--->kernel"<<kernel_id<<"\n";
        // /* iterate on basic block and inject the first instruction */
        // for (auto &bb : cfg.bbs) {
        //      cout <<"BB"<<count<<"\n";
        //      for (auto &i : bb->instrs) {
        //          i->print(" ");
        //      }
        //     bb_id++;
        // }

        const vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        uint32_t cnt = 0;
        /* iterate on the static instructions */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }

            if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = string(instr->getOpcode());
            }
            int opcode_id = opcode_to_id_map[instr->getOpcode()];
            int is_glob_loc = 0;
            int pred_num = -1;
            int mref_id = 0;
            int dst_oprnd = -1;
            int dst_oprnd_type = -1;
            int src_oprnds[5] = {-1};
            int src_oprnds_type[5] = {-1};     
            /*
            operands types:
                1 = REG & UREG
                2 = PRED & UPRED
                3 = MREF
            // ignore immediate and CBANK since it will not affect
            // the dependency check when printing out the instruction
            // predciates are also resolved at runtime, so we don't care much about but leave it for now
            */
            

            /* for cache memories */
            if( instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL ||
                instr->getMemorySpace() == InstrType::MemorySpace::LOCAL  ||
                instr->getMemorySpace() == InstrType::MemorySpace::GENERIC  ){
                is_glob_loc = 1;
            }

            if(instr->hasPred()){
                pred_num = (int)instr->getPredNum();
            }

            

            /* insert call to the instrumentation function with its arguments */
            nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
            /* predicate value */
            nvbit_add_call_arg_guard_pred_val(instr);
            /* programm counter */
            nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
            /* opcode id */
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            /* global or local mem. instruction? */
            nvbit_add_call_arg_const_val32(instr, is_glob_loc);
            /* memory reference 64 bit address */
            nvbit_add_call_arg_mref_addr64(instr, mref_id);

            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t *op = instr->getOperand(i);
                
                if (i == 0){ //handle dest oprnd
                    if(op->type == InstrType::OperandType::REG 
                        || op->type == InstrType::OperandType::UREG){ //dest oprnd is register 
                        dst_oprnd = op->u.reg.num;
                        dst_oprnd_type = 1;
                    } else if (op->type == InstrType::OperandType::PRED
                        || op->type == InstrType::OperandType::UPRED){ //1 oprnd is const immediate UINT64
                        dst_oprnd = op->u.pred.num;
                        dst_oprnd_type = 2;
                    } else if (op->type == InstrType::OperandType::MREF){ //dest oprnd is memory (i.e.: ST or REG)
                        if (is_glob_loc){
                            mref_id ++;
                        }
                        dst_oprnd_type = 3;
                        if(op->u.mref.has_ra){
                            // e.g., STG.E.64.SYS [R8], R6; [R8]  is Return Address Register
                            //       STS.64 [R69], R58 ;    [R69] is Return Address Register
                            dst_oprnd = op->u.mref.ra_num;
                        }else if(op->u.mref.has_ur){
                            // may unified addressing mode, rarely encountered such a situation
                            dst_oprnd = op->u.mref.ur_num;
                        }
                    }
                }else{ //handle src oprnds
                    if(op->type == InstrType::OperandType::REG 
                        || op->type == InstrType::OperandType::UREG){
                        src_oprnds[i] = op->u.reg.num;
                        src_oprnds_type[i] = 1;
                    } else if (op->type == InstrType::OperandType::PRED
                        || op->type == InstrType::OperandType::UPRED){
                        src_oprnds[i] = op->u.reg.num;
                        src_oprnds_type[i] = 2;
                    } else if (op->type == InstrType::OperandType::MREF){
                        if (is_glob_loc){
                            mref_id ++;
                        }
                        src_oprnds_type[i] = 3;
                        if(op->u.mref.has_ra){
                            src_oprnds[i] = op->u.mref.ra_num;
                        }else if(op->u.mref.has_ur){
                            src_oprnds[i] = op->u.mref.ur_num;
                        }
                    }
                }
            }

            /* memory references */
            nvbit_add_call_arg_const_val32(instr, mref_id);
            /* handle LDGSTS instruction with 2 memory references */
            if(mref_id == 2){
                nvbit_add_call_arg_mref_addr64(instr, 1);
            }else{
                nvbit_add_call_arg_mref_addr64(instr, 0);
            }

            /* destination operand */
            nvbit_add_call_arg_const_val32(instr, dst_oprnd);
            nvbit_add_call_arg_const_val32(instr, dst_oprnd_type);

            /* source operands */
            nvbit_add_call_arg_const_val32(instr, src_oprnds[0]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds_type[0]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds[1]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds_type[1]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds[2]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds_type[2]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds[3]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds_type[3]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds[4]);
            nvbit_add_call_arg_const_val32(instr, src_oprnds_type[4]);

            /* predicate num */
            nvbit_add_call_arg_const_val32(instr, pred_num);

            nvbit_add_call_arg_const_val32(instr, (int)(instr->isPredNeg()));
            nvbit_add_call_arg_const_val32(instr, (int)(instr->isPredUniform()));

            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);

            cnt++; 
        }
    }
}

#include <algorithm>
#include <iterator>

void dump_issue_config(){
    vector<int> issued_sms_num;

    issue_config_fp << "-trace_issued_sms_num " << SMid_CTAid_Map.size() << "\n";
    
    int idx1 = 0;
    for (auto it_map_issue = SMid_CTAid_Map.begin(); it_map_issue != SMid_CTAid_Map.end(); it_map_issue++) {
        
        if (issued_sms_num.size() < 80) {// TO DO: replace 70 by confiure
            auto iter = find(issued_sms_num.begin(), issued_sms_num.end(), it_map_issue->first);
            /* Don't find it_map_issue->first in issued_sms_num, need to add it. */
            if (iter == issued_sms_num.end()) {
                issued_sms_num.push_back(it_map_issue->first);
            }
        }
        
        issue_config_fp << "-trace_issued_sm_id_" << it_map_issue->first << " " 
                        << it_map_issue->second.size() << ","
                        << it_map_issue->first << ",";
        
        // cout << it_map_issue->first << " " << SMid_CTAid_timestamp_Map[it_map_issue->first].size() << endl;
         
        int idx2 = 0;
        for (auto it_tuple = it_map_issue->second.begin(); it_tuple != it_map_issue->second.end(); it_tuple++) {

            issue_config_fp << "(" << get<0>(*it_tuple) << ",";
            // issue_config_fp << get<1>(*it_tuple) << ",";
            // issue_config_fp << get<2>(*it_tuple) << ",";
            // issue_config_fp << get<3>(*it_tuple) << "),";
            /* calculate an index for the block the current instn belong to */
            int index = get<3>(*it_tuple) * kernel_gridY * kernel_gridX + kernel_gridX * get<2>(*it_tuple)  + get<1>(*it_tuple);

            // cout << "    " << it_map_issue->first << " " << idx2 << endl;
            uint64_t timestamp = SMid_CTAid_timestamp_Map[it_map_issue->first][idx2];

            issue_config_fp << index 
                            << "," << hex << timestamp << dec// ????????????????????????????
                            << "),";
            idx2++;
        }
        issue_config_fp << "\n";
        idx1++;
    }

    issue_config_fp << "-trace_issued_sms_vector "; 
    for (auto sm_num : issued_sms_num)
        issue_config_fp << sm_num << ",";
    issue_config_fp << endl;

    issue_config_fp.close();
}


void dump_instn_config(){
    for (int kernel = 1; kernel < kernel_id; kernel++){
        for (int pc = 0; pc < 10000000; pc += 16) {
            bool found = false;
            for (const auto& item : PCid_Instn_Map) {
                if (get<0>(item.first) == kernel && get<1>(item.first) == pc) {
                    instn_config_fp << item.second << endl;
                    found = true;
                }
            }
            if (!found) {
                break;
            }
        }
    }
    instn_config_fp.close();
}

__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is completed */
    inst_access_t ma;
    ma.cta_id_x = -1;
    channel_dev.push(&ma, sizeof(inst_access_t));
    // /* flush channel */
    channel_dev.flush();
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                        const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    if (kernel_id > MAX_KERNELS){
        exit(0);
    } 

    if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {

        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

            if (!is_exit) {
                pthread_mutex_lock(&mutex);
                instrument_function_if_needed(ctx, p->f);
                nvbit_enable_instrumented(ctx, p->f, true);
                recv_thread_receiving = true;

                cout << "Starting kernel #" << kernel_id << "...\n";

                kernel_gridX = p->gridDimX;
                kernel_gridY = p->gridDimY;
                kernel_gridZ = p->gridDimZ;

                string file_name = "./sass_traces/kernel_"+ to_string(kernel_id) + ".sass";
                insts_trace_fp.open(file_name);
            } else {
                /* make sure current kernel is completed */
                cudaDeviceSynchronize();
                assert(cudaGetLastError() == cudaSuccess);

                /* make sure we prevent re-entry on the nvbit_callback when issuing
                * the flush_channel kernel */
                skip_flag = true;

                /* issue flush of channel so we are sure all the accesses
                * have been pushed */
                flush_channel<<<1, 1>>>();
                cudaDeviceSynchronize();
                assert(cudaGetLastError() == cudaSuccess);

                /* unset the skip flag */
                skip_flag = false;
                
                /* wait here until the receiving thread has not finished with the current kernel */
                while (recv_thread_receiving) {
                    pthread_yield();
                }

                int gridX = 0, gridY = 0, gridZ = 0, blockX = 0, blockY = 0, blockZ= 0,\
                nregs=0, shmem_static_nbytes=0, shmem_dynamic_nbytes = 0, stream_id = 0;

                CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));
                CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

                gridX = p->gridDimX;
                gridY = p->gridDimY;
                gridZ = p->gridDimZ;
                blockX = p->blockDimX;
                blockY = p->blockDimY;
                blockZ = p->blockDimZ;
                stream_id = (uint64_t)p->hStream;
                shmem_dynamic_nbytes = p->sharedMemBytes;

                int num_ctas = gridX * gridY * gridZ;

                int threads_per_cta = blockX * blockY * blockZ;
                int tot_num_thread = num_ctas * threads_per_cta;
                int tot_num_warps =  tot_num_thread/32;
                if(tot_num_warps ==0)
                    tot_num_warps = 1;

                string kernel_name = nvbit_get_func_name(ctx, p->f);
                std::cout << kernel_name << endl;
                string delimiter = "(";
                kernel_name  = kernel_name.substr(0, kernel_name.find(delimiter));
                replace(kernel_name.begin(), kernel_name.end(), ' ', '_');

                app_config_fp << "-kernel_" + to_string(kernel_id) << "_kernel_name " << kernel_name << "\n"; // BUG
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_num_registers " << nregs << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_shared_mem_bytes " 
                              << (shmem_static_nbytes + shmem_dynamic_nbytes) << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_grid_size " << num_ctas << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_block_size " << threads_per_cta << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_cuda_stream_id " << stream_id << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_grid_dim_x " << gridX << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_grid_dim_y " << gridY << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_grid_dim_z " << gridZ << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_tb_dim_x " << blockX << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_tb_dim_y " << blockY << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_tb_dim_z " << blockZ << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_shmem_base_addr " 
                              << (uint64_t)nvbit_get_shmem_base_addr(ctx) << "\n";
                app_config_fp << "-kernel_" + to_string(kernel_id) << "_local_base_addr " 
                              << (uint64_t)nvbit_get_local_mem_base_addr(ctx) << "\n";
                
                cout << "Exiting kernel #" << kernel_id << "...\n";
                kernel_id++;
                first_warp_exec = 0;
                first_kernel_mem_clk = 0;
                inst_count = 0;
                reg_dependency_map.clear();
                pred_dependency_map.clear();
                insts_trace_fp.close();
                
                pthread_mutex_unlock(&mutex);
            }
    }
}

uint64_t kernel_mem_clk = 0;

void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;

        if (recv_thread_receiving && 
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {

                inst_access_t *ia = (inst_access_t *)&recv_buffer[num_processed_bytes];

                if (ia->cta_id_x == -1) {
                    recv_thread_receiving = false;
                    break;
                }
                if (first_warp_exec == 0){
                    current_sm_id = ia->sm_id;
                    current_cta_id_x = ia->cta_id_x;
                    current_cta_id_y = ia->cta_id_y;
                    current_cta_id_z = ia->cta_id_z;
                    current_warp_id = ia->warp_id;
                    first_warp_exec = 1;
                }

                // if(ia->sm_id == current_sm_id && ia->cta_id_x == current_cta_id_x && ia->cta_id_y == current_cta_id_y && ia->cta_id_z == current_cta_id_z){
                        /* the SM id that every CTA is issued to */
                        auto it_map_smid_ctaid = SMid_CTAid_Map.find(ia->sm_id);                                          // yangjianchao16 add
                        auto it_map_smid_cta_id_timestamp = SMid_CTAid_timestamp_Map.find(ia->sm_id);                     // yangjianchao16 add
                        auto item_tuple = make_tuple(kernel_id, ia->cta_id_x, ia->cta_id_y, ia->cta_id_z);                // yangjianchao16 add
                        if (it_map_smid_ctaid == SMid_CTAid_Map.end()){ // the first time sm_id occurs                    // yangjianchao16 add
                            SMid_CTAid_Map[ia->sm_id].push_back(item_tuple);                                              // yangjianchao16 add
                            SMid_CTAid_timestamp_Map[ia->sm_id].push_back(ia->curr_clk);                                  // yangjianchao16 add
                        } else { // not the first time sm_id occurs                                                       // yangjianchao16 add
                            if (find(it_map_smid_ctaid->second.begin(), it_map_smid_ctaid->second.end(), item_tuple) ==   // yangjianchao16 add
                                it_map_smid_ctaid->second.end()) {                                                        // yangjianchao16 add
                                it_map_smid_ctaid->second.push_back(item_tuple);                                          // yangjianchao16 add
                                it_map_smid_cta_id_timestamp->second.push_back(ia->curr_clk);                             // yangjianchao16 add
                            }                                                                                             // yangjianchao16 add
                        }                                                                                                 // yangjianchao16 add

                        /* the pc that every instn actually executed on the fly belongs to */
                        auto it_map_pcid_instn = PCid_Instn_Map.find(make_tuple(kernel_id, ia->pc));                      // yangjianchao16 add
                        if (it_map_pcid_instn == PCid_Instn_Map.end()){ // the first time <kernel_id, pc> occurs          // yangjianchao16 add
                            string Instn_string = "";                                                                     // yangjianchao16 add
                            /* kernel_id, pc */
                            stringstream ss;                                                                         // yangjianchao16 add
                            ss << hex << ia->pc;                                                                          // yangjianchao16 add
                            Instn_string += to_string(kernel_id) + " " + ss.str() + " ";                                  // yangjianchao16 add
                            /* pred */
                            if (ia->pred_num != -1) {                                                                     // yangjianchao16 add
                                if (ia->isPredNeg)                                                                        // yangjianchao16 add
                                    Instn_string += "@!P" + to_string(ia->pred_num) + " ";                                // yangjianchao16 add
                                else                                                                                      // yangjianchao16 add
                                    Instn_string += "@P" + to_string(ia->pred_num) + " ";                                 // yangjianchao16 add
                            }                                                                                             // yangjianchao16 add
                            /* opcode */
                            Instn_string += id_to_opcode_map[ia->opcode_id] + " ";                                        // yangjianchao16 add
                            /* destination operands */
                            if (ia->dst_oprnd_type == 1){                                                                 // yangjianchao16 add
                                Instn_string += "R" + to_string(ia->dst_oprnd) + " ";                                     // yangjianchao16 add
                            } else if (ia->dst_oprnd_type == 2){                                                          // yangjianchao16 add
                                Instn_string += "P" + to_string(ia->dst_oprnd) + " ";                                     // yangjianchao16 add
                            } else if (ia->dst_oprnd_type == 3){                                                          // yangjianchao16 add
                                Instn_string += "[R" + to_string(ia->dst_oprnd) + "] ";                                   // yangjianchao16 add
                            }                                                                                             // yangjianchao16 add
                            /* src operands */
                            for (int i = 0; i < 5; i++){                                                                  // yangjianchao16 add
                                if (ia->src_oprnds[i] != -1){                                                             // yangjianchao16 add
                                    if (ia->src_oprnds_type[i] == 1 || ia->src_oprnds_type[i] == 3){                      // yangjianchao16 add
                                        Instn_string += "R" + to_string(ia->src_oprnds[i]) + " ";                         // yangjianchao16 add
                                    } else if (ia->src_oprnds_type[i] == 2){                                              // yangjianchao16 add
                                        Instn_string += "P" + to_string(ia->src_oprnds[i]) + " ";                         // yangjianchao16 add
                                    }                                                                                     // yangjianchao16 add
                                }                                                                                         // yangjianchao16 add
                            }                                                                                             // yangjianchao16 add
                            PCid_Instn_Map[make_tuple(kernel_id, ia->pc)] = Instn_string;                                 // yangjianchao16 add
                        }                                                                                                 // yangjianchao16 add

                        // insts_trace_fp << "PC-0x" << hex << ia->pc << dec << " ";
                        // insts_trace_fp << "CLK-0x" << hex << ia->curr_clk << dec << " ";
                        // insts_trace_fp << "sm_id-" << ia->sm_id << " " 
                        //                << "cta_id_x-" << ia->cta_id_x << " " 
                        //                << "cta_id_y-" << ia->cta_id_y << " " 
                        //                << "cta_id_z-" << ia->cta_id_z << " " 
                        //                << "warp_id-" << ia->warp_id<<" "
                        //                << "gwarp_id-" << ia->gwarp_id<<" ";
                        
                        insts_trace_fp << hex << ia->pc << dec << " " << ia->gwarp_id << " ";

                        // /* opcode */
                        // insts_trace_fp<<id_to_opcode_map[ia->opcode_id]<<" ";

                        // /* destination operands */
                        // if(ia->dst_oprnd_type == 1 || ia->dst_oprnd_type == 3){
                        //     insts_trace_fp<<"R"<<ia->dst_oprnd << " ";
                        // }else if (ia->dst_oprnd_type == 2){
                        //     insts_trace_fp<<"P"<<ia->dst_oprnd << " ";
                        // }
                        
                        // /* src operands */
                        // for(int i=0; i<5; i++){
                        //     if(ia->src_oprnds[i] != -1){
                        //         if (ia->src_oprnds_type[i] == 1 || ia->src_oprnds_type[i] == 3){
                        //             insts_trace_fp<<"R"<<ia->src_oprnds[i]<<" ";
                        //         }else if(ia->src_oprnds_type[i] == 2){
                        //             insts_trace_fp<<"P"<<ia->src_oprnds[i]<< " ";
                        //         }
                        //     }
                        // }

                        // insts_trace_fp<<"\n";
                // }


                // bitset<32> mask(ia->active_mask & ia->predicate_mask);
                if (ia->is_mem_inst == 1){
                    if (first_kernel_mem_clk == 0) {
                        kernel_mem_clk = ia->curr_clk;
                        first_kernel_mem_clk = 1;
                    }

                    ofstream mem_trace_fp;

                    /* calculate an index for the block the current mem reference belong to */
                    int index = ia->cta_id_z * kernel_gridY * kernel_gridX + kernel_gridX * ia->cta_id_y  + ia->cta_id_x;

                    string file_name = "./memory_traces/kernel_"+ to_string(kernel_id) + "_block_"+to_string(index)+".mem";
                    mem_trace_fp.open(file_name, ios::app);
                    
                    // mem_trace_fp << "\n=====\n";
                    // mem_trace_fp << id_to_opcode_map[ia->opcode_id] << " ";
                    
                    mem_trace_fp << hex << ia->pc << " ";

                    // mem_trace_fp << id_to_opcode_map[ia->opcode_id] << " ";

                    // mem_trace_fp << hex << (ia->active_mask & ia->predicate_mask) << " ";
                    
                    mem_trace_fp << hex << int(ia->curr_clk) - int(kernel_mem_clk) << " ";
                    // for (int m = 0; m < 32; m++) {
                    //     if(ia->mem_addrs1[m]!=0){
                    //         mem_trace_fp<<"0x"<<hex<<ia->mem_addrs1[m]<<" ";
                    //     } 
                    // }
                    mem_trace_fp << hex << ia->mem_addrs1[0] << " ";
                    // if (ia->mref_id == 2){
                    //     for (int m = 0; m < 32; m++) {
                    //         if(ia->mem_addrs2[m]!=0){
                    //             mem_trace_fp << hex << ia->mem_addrs2[m]<<" ";
                    //         } 
                    //     }
                    // }
                    if (ia->mref_id == 2) mem_trace_fp << hex << ia->mem_addrs2[0] << " ";
                    mem_trace_fp << endl;
                    mem_trace_fp.close();
                }
                
                num_processed_bytes += sizeof(inst_access_t);
            }  
        } 
    }
    free(recv_buffer);
    return NULL;
}


void nvbit_at_ctx_init(CUcontext ctx) {
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}


void nvbit_at_ctx_term(CUcontext ctx) {
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
    }

    dump_app_config();
    dump_issue_config();
    dump_instn_config();
}
