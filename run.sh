directory=$1
timelimit=$2
for f in `ls $1`; do
    # echo $1/$f;
    cmd="/home/chuwen/anaconda3/bin/python -u main.py --fp $1/$f --verbosity 1 --time_limit $2 &> $f.log"
    echo $cmd;
done