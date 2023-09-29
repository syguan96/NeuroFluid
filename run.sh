OUTPUT_SCENES_DIR=$1
mkdir $OUTPUT_SCENES_DIR
for seed in `seq 1 1`; do
        python create_physics_scenes.py \
                --output $OUTPUT_SCENES_DIR \
                --seed $seed \
                --default-fluid \
                --default-box \
                --default-viscosity \
                --default-density \
                --default-vel \
                --obj2volume_scale 1.0
done 


dirs=`ls $OUTPUT_SCENES_DIR`
for dir in $dirs; do
    mkdir $OUTPUT_SCENES_DIR/$dir/mesh
    python create_surface_meshes.py \
           --input_glob "$OUTPUT_SCENES_DIR/$dir/output/fluid*.npz" \
           --outdir $OUTPUT_SCENES_DIR/$dir/mesh/
done

