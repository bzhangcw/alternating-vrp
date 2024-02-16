wdir="dataset/solomon-100-original"
hyperparams="--sigma 1.8 --tsig 1.5 --rho0 1"
output="output-$(date +"%Y%m%d%H%M")"

header=$1
mkdir -p $output
if [ -f cmd.sh ]; then
    rm cmd.sh
fi

# vehicles, customers
# for c100
scales=(
  "3,25"
  "5,50"
  "10,100"
)
# for c2
# scales=(
#   "3,100"
# )

for size in $scales; do
  IFS=',' read nv nc <<< "${size}"

  for ff in $(ls $wdir/$header*.txt); do
    fname=$(basename -s .txt $ff)
    cmd="python -u main_bcd.py --fp $ff ${hyperparams} --n_vehicles $nv --n_customers $nc --output $output/$fname.$nv-$nc.json --time_limit 200 &> $output/$fname.$nv-$nc.log"
    echo $cmd >> cmd.sh
  done
done

echo "command saving to cmd.sh"
echo "create ./$output for saving solutions..."

