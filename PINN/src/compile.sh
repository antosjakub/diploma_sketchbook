

#t_list=("0" "1" "3" "4")
#t_list=("2" "5" "6" "7")

#t_list=("0" "1" "4")
t_list=("5" "6" "7")

#for t in $(seq 0 7); do
for t in "${t_list[@]}"; do
    #echo $t
    python compile.py $t
done
