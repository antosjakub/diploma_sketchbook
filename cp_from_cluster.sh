

#cluster_path=simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq
cluster_path=simulant_antos@tarkil.metacentrum.cz:/storage/praha1/home/simulant_antos/diploma_sketchbook/PINN/case0_HeatEq

#dir_path=gridsearch__26-07-11--15:32:36__GS_d6_bs2048
#dir_path=gridsearch__26-07-11--22:35:08__GS_d6_bs1024

#dir_path="gridsearch__26-07-14--01:23:28__pinn_overnight_d6_rbas_k=1.0"
#dir_path="gridsearch__26-07-14--11:54:02__pinn_overnight_d6_rbas_k=2.0"
#dir_path="gridsearch__26-07-15--15:15:37__pinn_overnight_d6_rbas_k=0.5"


#dir_path="gridsearch__26-07-15--21:01:49__pinn_d8_100k_4x512_rbas"
#dir_path="gridsearch__26-07-16--05:19:49__pinn_d8_160k_4x256"
#dir_path="gridsearch__26-07-16--05:52:47__pinn_d8_160k_4x512"
#dir_path="gridsearch__26-07-16--06:11:46__pinn_d8_bs_80k_4x512_larger_bs"
dir_path="gridsearch__26-07-16--17:33:40__pinn_d8_bs2_80k_4x256"



here_path="cluster_d8"

rsync -avz "$cluster_path/$dir_path/" "$here_path/$dir_path/"


#rsync -avz --exclude='*.png'
#username@cluster_address:/path/to/source_folder /path/to/destination_folder 
#
#rsync -avz --exclude='*.png' simulant_antos@tarkil.metacentrum.cz:~/diploma_sketchbook/PINN/case0_HeatEq/gridsearch__26-07-08--10:11:39__GS_6_bs_and_resample/ cluster_res/