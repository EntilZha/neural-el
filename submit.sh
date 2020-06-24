#!/bin/bash
for i in {0..2300}
do
start=$((i * 50))
end=$(((i + 1) * 50))
echo "$start $end"
c="python3 neuralel.py data/qanta.train.2018.04.18.json data/output_${start}_${end}.json $start $end"
jobname="nel_${start}_${end}"
echo "#!/bin/bash
$c" > submit2.sh
sbatch -e "runs/${jobname}_error.txt" -o "runs/${jobname}_output.txt" --qos=gpu-short --partition=gpu --gres=gpu:1 --cpus-per-task=2 --time=2:00:00 --mem=60g --exclude=materialgpu00,materialgpu01 submit2.sh
done



