# Intro to Machine Learning - TensorFlow Project

Project code for Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. In this project, you will first develop code for an image classifier built with TensorFlow, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.

### Submission Explaination

####Part1: The directory contains a HTML format file to look through for part 1, or you can view the notebook directly.

####Part2: There will be such possible workflows:

1. You have to include the image_path and the tensorflow model in the format of:

```
Python predict.py image_path saved_model
```

You can choose them from ./test_images and ./models folder.

2. If you include the "--top_k" parameter in the format of:

```
Python predict.py image_path saved_model --top_k K
```

You can view the top k possiblity

3. If you also want to know the real class name but not the label:

```
Python predict.py image_path saved_model [--top_k K] --category_names map
```

Where you can find the class name map from './label_map.json'