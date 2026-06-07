# Computer Vision 2026 Final Project

## Final Project

The Final Project is a valuable opportunity for you to apply the concepts and techniques learned throughout the course to a computer vision problem of your own interest. We highly encourage you to work in teams— each team may consist of up to four members.

You are free to choose from the suggested project topics listed below. Topics 1–3 focus on classical computer vision tasks, while topics 4–6 explore more advanced and cutting-edge research directions. To encourage exploration of frontier areas, projects based on topics 4–6 will be eligible for 10 bonus points, meaning the maximum project score will be 110 instead of the standard 100.

We recommend selecting a topic that not only challenges your technical skills but also aligns with your interests and future goals.

## Project Topics

### 1. Single object tracking

The single object tracking system is a technology capable of matching a target that is given by a person in the first frame of a video with the subsequent video sequence. This kind of system is usually used to security monitor, and captures the target by calculating the similarity between the reference frame target and the search frame target.

Task: Collect multiple videos by yourself, use the labeling tool to mark the object that needs to be tracked in the first frame, design tracking algorithm to track and visualize the tracking results.

Survey:

https://blog.csdn.net/qq_37002417/article/details/108141409

https://zhuanlan.zhihu.com/p/503735985

Dataset: http://got-10k.aitestunion.com/

Github:

https://github.com/visionml/pytracking

https://github.com/heartexlabs/labellmg

### 2. Semantic Segmentation

Semantic segmentation is a typical computer vision problem that involves taking as input some raw data and converting them into a mask with highlighted regions of interest, with applications such as scene understanding, medical image analysis, robotic perception, video surveillance, augmented reality, and image compression, among many others.

Task: Train a semantic segmentation model(not limited to the PASCAL VOC dataset) and collect some interesting scenarios yourself to see how well the segmentation works. Also, we encourage you to try your model for image segmentation on videos and see what difficulties you get.

Dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

Github: https://github.com/usuyama/pytorch-unet (This is a relatively simple method, and you are free to choose the method you like)

### 3. Instance Segmentation

Instance segmentation is a core task in computer vision that bridges object detection and semantic segmentation. It requires the model to accomplish two goals simultaneously: first, recognize the categories of all objects of interest in the image and locate their bounding boxes (object detection); second, generate precise pixel-level segmentation masks for each individual object instance.

Project Tasks

Select a classic instance segmentation model for reproduction or secondary development based on open- source frameworks. Complete model training and validation on the COCO 2017 instance segmentation dataset or the PASCAL VOC 2012 extended dataset, and implement the model inference function to output visual results including category labels, bounding boxes and instance segmentation masks for input images. Evaluate the model performance on the test set of the corresponding dataset and report the following standard metrics.

Datasets and Reference Resources

Main Datasets:

COCO 2017 Instance Segmentation Dataset: https://cocodataset.org/#download

PASCAL VOC 2012 Instance Segmentation Extension: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/i ndex.html

Medical Imaging Example Dataset: ISIC 2018 Skin Lesion Segmentation https://challenge.isic-archive.co m/landing/2018/

Reference Codes and Tools:

Detectron2 Official Implementation (including Mask R-CNN): https://github.com/facebookresearch/dete ctron2

YOLOv8-seg Official Implementation: https://github.com/ultralytics/ultralytics

Image Annotation Tool: LabelMe https://github.com/wkentaro/labelme

Evaluation Tool: pycocotools https://github.com/cocodataset/cocoapi

Core Paper References:

Mask R-CNN (ICCV 2017): https://arxiv.org/abs/1703.06870

YOLOv8: You Only Look Once Version 8: https://github.com/ultralytics/ultralytics

### 4. Open-Vocabulary Object Detection and Visual Grounding

Open-vocabulary object detection aims to detect objects using arbitrary text descriptions rather than a fixed category set. Visual grounding is a related task that localizes the image region described by a natural- language expression, such as "the red car on the left" or "the person holding an umbrella".

This project is slightly more challenging than classical topics such as face recognition, tracking, and semantic segmentation, because it requires both visual recognition and language understanding. Students may start from existing open-source models and focus on reproduction, evaluation, and analysis.

Project Tasks

This project consists of three main components.

Method Reproduction

Reproduce an open-vocabulary detection or visual grounding pipeline using an existing model.

Recommended models:

Grounding DINO

GLIP

OWL-ViT

YOLO-World

Detic

Dataset Evaluation

Evaluate the model on at least one public dataset or a meaningful subset.

Recommended datasets:

COCO

LVIS

ODinW

RefCOCO / RefCOCO+ / RefCOCOg

Flickr30K Entities

Resources

Github:

Grounding DINO: https://github.com/IDEA-Research/GroundingDINO

GLIP: https://github.com/microsoft/GLIP

YOLO-World: https://github.com/AILab-CVC/YOLO-World

Detic: https://github.com/facebookresearch/Detic

HuggingFace:

OWL-ViT: https://huggingface.co/docs/transformers/model_doc/owlvit

Datasets:

COCO: https://cocodataset.org/

LVIS: https://www.lvisdataset.org/

RefCOCO: https://github.com/lichengunc/refer

Flickr30K Entities: http://shannon.cs.illinois.edu/DenotationGraph/

Evaluation:

COCO API: https://github.com/cocodataset/cocoapi

pycocotools: https://github.com/ppwwyyxx/cocoapi

### 5. Segmentation in 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) has garnered significant attention in the field of 3D reconstruction, novel view synthesis, and related domains due to its efficient and differentiable rendering capabilities. Notably, Object- Level Segmentation within 3DGS plays a pivotal role in various downstream applications, including scene editing, scene understanding, and embodied intelligence.

Project Tasks

This project consists of two main components:

1. Method Reproduction

Object Segmentation: Perform object-level segmentation on 3DGS scenes.

Downstream Applications: Based on the segmented objects, explore and implement several possible tasks, including:

3D Object Removal

3D Object Inpainting

3D Object Style Transfer

3D Multi-Object Editing

Or other creative and meaningful applications you come up with

Dataset Requirement: Conduct experiments on at least three different scenes from the provided datasets.

2. Your Contribution

Custom Data Validation: Capture and use your own real-world data to test the same pipeline.

Method Enhancements: Make any improvements or modifications to the segmentation process or the downstream applications. This could be as simple as parameter tuning for a specific scenario—novel ideas are welcome but not required.

Note: The emphasis of this project lies in the process of trying and validating, rather than achieving optimal results. Recourse:

Dataset:

LERF-MASK: https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md

Mip-NeRF360: https://jonbarron.info/mipnerf360/

Github: https://github.com/lkeab/gaussian-grouping (Free to choose other methods.)

3DGS Viewer: https://superspl.at/editor

### 6. Multi-view stereo SLAM

Multi-view stereo (MVS) extends the principles of passive stereo to multiple viewpoints and aims to reconstruct a dense 3D model of a scene from a sequence of images with known camera parameters. Multi-view stereo SLAM extends this progress, to Incrementally estimate camera parameters while reconstructing.

Task: In this project, you need to run a MVS-SLAM pipeline. accepting monocular video as input, and output 3D scene point cloud. Test your result in public dataset, and build a live demo using SUSTech scene (classrooms or laboratories)

Bonus: you can consider use NVS methods (like 3d gaussian splatting) to build a better visual effect of your result.

Datasets:

7-Scenes: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/ (copy the download link and paste to search bar to download it if you meet problem).

Github: https://github.com/rmurai0610/MASt3R-SLAM

Note: We do not limit the datasets and github repositories you use, you can use the resources you find to complete the task.

## Forming Groups

Groups can have up to 4 members.

Only one report should be submitted per group, and it should include a section highlighting the contributions of each team member.

Please complete the topic selection and team member information on Tencent Docs: https://docs.qq.c om/sheet/DTnJRUGpIZXdYdk9S?tab=BB08J2VFZRXFV?tab=BB08J2. The deadline for grouping is 2026.5.31.

## Report

We’ve provided a template to help guide your final project report, but feel free to follow your own structure as long as it is clear and well-organized. Latex is also recommended for your report.

Regarding the reports:

Each group should submit one report.

The report should include the names of all the collaborators.

You can use word, markdown, or Latex to form your report.

PDF file should be submitted.

You should describe and evaluate what you did in your project, which may not necessarily be what you hoped to do originally. A small result described and evaluated well will earn more credit than an ambitious result where no aspect was done well. Be accurate in describing the problem you tried to solve. Explain in detail your approach, and specify any simplifications or assumptions you have taken. Also demonstrate the limitations of your approach. When doesn't it work? Why? What steps would you have taken have you continued working on it? Make sure to add references to all related work you reviewed or used.

You are allowed to submit any supplementary material that you think it important to evaluate your work, however we do not guarantee that we will review all of that material, and you should not assume that. The report should be self-contained.

Submission: submit your report to blackboard as a pdf file named groupid_final.pdf. Submit any supplementary material(e.g. videos) as a single zip file named groupid_sup.zip. Add a README file describing the supplemental content. Submit your code as single zip file named groupid_code.zip.

## Grading Policy

Report (60)

Introduction (10)

Related work (5)

Approach (10)

Experimental results (20)

Conclusion (5)

References (5)

Overall clarity of the report (5)

Contribution of each member (e.g.35% for xxx,25% for xxx): we will add or deduct

Presentation (40)

15 min per group + 3 min Q&A

Topic choice bonus(10): As long as you choose a topic from 4 to 8 and complete the project in a satisfactory manner, you will receive 10 bonus points. However, if the quality of the work is too low, the bonus points will not be awarded.

## FAQ

Q: Does this major assignment require innovation in the algorithm?

A: We highly encourage students to innovate in their algorithms. However, due to time constraints, it is not mandatory. You can treat it more as an application-oriented project and focus on demonstrating your engineering efforts and teamwork. Of course, if any student does come up with innovative algorithms that perform well on standard datasets, we will award extra points for that.

Q: How will TA determine the contribution level?

A: For groups that have open-sourced on GitHub, we can roughly judge the contribution level based on commits. However, we mainly rely on the contributions section in your reports.

