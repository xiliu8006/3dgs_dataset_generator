#!/bin/bash

# 参数解析函数
parse_args() {
    while [ $# -gt 0 ]; do
        case $1 in
            --data_root)
                DATA_ROOT="$2"
                shift
                ;;
            --output_dir)
                OUTPUT_DIR="$2"
                shift
                ;;
            --num_samples)
                NUM_SAMPLES=()
                while [[ "$2" =~ ^[0-9]+$ ]]; do
                    NUM_SAMPLES+=("$2")
                    shift
                done
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
        shift
    done
}

# 参数解析
parse_args "$@"

# 确保脚本目录存在
SCRIPTS_DIR="scripts"
mkdir -p "$SCRIPTS_DIR"

# num_samples 转换为字符串
NUM_SAMPLES_STR=$(printf " %s" "${NUM_SAMPLES[@]}")
NUM_SAMPLES_STR=${NUM_SAMPLES_STR:1}

# 遍历场景并提交 sbatch 任务
for SCENE in "$DATA_ROOT"/*; do
    if [ -d "$SCENE" ]; then  # 确保是目录
        SCENE_NAME=$(basename "$SCENE")
        JOB_NAME="job-${SCENE_NAME}"

        # 创建一个以作业名称命名的 SLURM 脚本
        cat <<EOF > "$SCRIPTS_DIR/${JOB_NAME}.sh"
#!/bin/bash

#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32gb
#SBATCH --time 72:00:00
#SBATCH --gpus-per-node v100:1

source /etc/profile.d/modules.sh

# 加载必要的模块
module add cuda/11.8

# 激活环境
source activate freenerf

cd /scratch/chaoyiz/code/FreeNeRF

# 运行训练渲染
srun python dataset_generator/train_render.py --scene_path ${SCENE} --output_path ${OUTPUT_DIR}/${SCENE_NAME} --num_samples ${NUM_SAMPLES_STR} --use_lr

EOF

        # 提交作业
        sbatch "$SCRIPTS_DIR/${JOB_NAME}.sh"
    fi
done

#./generate_jobs.sh --data_root /path/to/data_root --output_dir /path/to/output_dir --num_samples 3 6 9
