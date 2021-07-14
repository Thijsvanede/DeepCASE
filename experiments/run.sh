#!/bin/bash

# Set defaults
data=../data/data_full.csv
max_default="1e6"
input_default=5
output_default=1
batch_default=32
epochs_default=10

max="1e4 1e5 1e6 1e7 1e8"
input="1 2 3 4 5 6 7 8 9 10"
output="1 2 3 4 5 6 7 8 9 10"
batch="1 2 4 8 16 32 128 256 512 1024"
epochs="1 5 10 20 50 100"


# Loop over max size
# for m in $max; do
#   i=${input_default}
#   o=${output_default}
#   b=${batch_default}
#   e=${epochs_default}
#   outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
#   # Show progress
#   echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
#   # Run python script
#   { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
# done

# Loop over input size
for i in $input; do
  m=${max_default}
  o=${output_default}
  b=${batch_default}
  e=${epochs_default}
  outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
  # Show progress
  echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
  # Run python script
  { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
done

# Loop over input size
for o in $output; do
  m=${max_default}
  i=${input_default}
  b=${batch_default}
  e=${epochs_default}
  outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
  # Show progress
  echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
  # Run python script
  { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
done

# Loop over input size
for b in $batch; do
  m=${max_default}
  i=${input_default}
  o=${output_default}
  e=${epochs_default}
  outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
  # Show progress
  echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
  # Run python script
  { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
done
# Loop over input size
for e in $epochs; do
  m=${max_default}
  i=${input_default}
  o=${output_default}
  b=${batch_default}
  outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
  # Show progress
  echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
  # Run python script
  { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
done


# # Loop over input size
# for m in $max; do
#   # Loop over input dimension
#   for i in $input; do
#     # Loop over output dimension
#     for o in $output; do
#       # Loop over batch size
#       for b in $batch; do
#         # Loop over epochs
#         for e in $epochs; do
#           # Show progress
#           echo "max=${m} input=${i} output=${o} batch=${b} epochs=${e}"
#           # Set output file
#           outfile="../results/output/data_full_dim_input_${i}_dim_output_${o}_max_${m}_batch_size_${b}_epochs_${e}.txt"
#           # Run python script
#           { time python3 seq2seq.py $data -m $m -i $i -o $o -b $b -e $e --silent ; } &> $outfile
#         done
#       done
#     done
#   done
# done
