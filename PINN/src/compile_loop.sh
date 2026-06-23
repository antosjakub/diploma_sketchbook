

compile_model_list=("0" "1")
train_type_list=("train_h" "train_ff" "train_fr" "train_rf" "train_rr")
#train_type_list=("train_h")


for compile_model in "${compile_model_list[@]}"; do
        for train_type in "${train_type_list[@]}"; do
            echo "=============================================================="
            #echo RUN: --compile_model=$compile_model --compile_train=$compile_train --train_type=$train_type
            echo RUN: --compile_model=$compile_model --train_type=$train_type
            python compile_loop.py --compile_model=$compile_model --train_type=$train_type
            python compile_loop.py --compile_model=$compile_model --train_type=$train_type
        done
done

echo "=============================================================="
python compile_loop.py --train_type=train_class
python compile_loop.py --train_type=train_class