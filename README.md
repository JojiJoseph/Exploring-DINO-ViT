# Exploring Dino ViT

Inspired from the paper [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814)

## Sample Usage

Use `python3 scriptname.py --help` to know about other arguments

To visualize pca

```bash
python3 visualize_pca.py --url https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
```

To visualize attention

```bash
python3 visualize_attention.py --url https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
```

To segment interactively, enter following on console and click somewhere in the image.

```bash
python3 interactive_segmentation.py --url https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
```

## TODO
    [] Replace Hugging Face DINOv1 models with that of facebook models.
