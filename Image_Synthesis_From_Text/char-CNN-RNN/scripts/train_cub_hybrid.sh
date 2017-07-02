
th train_sje_hybrid.lua \
  -data_dir pascal1k \
  -image_dir images \
  -ids_file trainids.txt \
  -learning_rate 0.0007 \
  -symmetric 1 \
  -max_epochs 200 \
  -savefile sje_cub_pascal_hybrid \
  -num_caption 10 \
  -gpuid 0 \
  -print_every 10

