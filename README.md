# Consistent Multi-Face Generation with SDXL
Default IP-Adapters with StableDiffusion XL model suffer from inability to properly localise effects of multiple IP-Adapters to multiple facial identities and keep the image variability the same. With this algorithmic pipeline approach, we solve this problem and introduce a Face-Matching module for selecting the most fitting position for each reference face in the final image.

The algorithm consists of 
1. Generation of Reference image (with sdxl or any other model)
2. Face Detection on the generated reference image
3. Face Matching between generated faces and our reference faces (Used for IP-Adapters)
4. Contructing a depth map from the reference image (used for depth ControlNet)
5. Generatin the main image using IP-Adapters localised on the bounding boxes of matching faces (attention masks), and ControlNet with depth image
6. Improving the Facial Features of each subject by Facial Inpainting and IP-Adapters (now without attention masks)

## Experimentation
To run the experiments, Comfy backend for stable diffusion image generation is needed (24GB Vram Nvidia GPU), running locally or in the cloud.
Setup the `.env` file with variables:
```
COMFY_HOST=127.0.0.1 (Or the actual IP from the cloud)
COMFY_PORT=8888 (Or the portu you've specified
```

Add the root dir to PYTHONPATH.

Then run 
- `python image_creation/create_identities.py` to create the reference identities
- `python image_creation/create_base_text_to_image.py` to create the baseline images
- `python image_creation/create_base_text_to_image.py` to create the multi-character images

To create all the images, the process might take a few days worth of generation as there are multiple configurations and each generates 500 images.

The method was evaluated on various metrics:
- Face Consistency
- Face Quality
- Head position Variability
- Image Qualtiy and Naturalness

First, face detection is needed in the generated images, so run:
`python face_detection/face_detection.py`

To evaluate in each metric, run the appropriate `main.py` file in the directories:
- `clip_iqa`
- `face_similarity`
- `face_quality`
- `headpose_extraction`

To calculate average scores, run the `metrics_visualization.ipynb` notebook.

It is also possible to view all the images and the corresponding calculated metrics per image by running
`streamlit run visualise/dataset_visualizer.py`
By using web interface, you can browse among all the generated images, show the detected bounding boxes, identities, face similarities, CLIP-IQA scores, and head poses.



