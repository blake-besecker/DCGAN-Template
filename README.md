anime_faces.py is the code for the model. face-generator is the trained model. It's trained with the anime-faces dataset from huggingface:
https://huggingface.co/datasets/huggan/anime-faces
It's trained from files locally. You'll need to download that set and import it into your local repo to train.
Or you can download the model already trained as I've provided, and generate as many fake people as you want!
It's worth noting that my resources are not necessarily the best, and the images my trained model generates is only 128x128, so not the highest resolution. It's perfectly fine and quite easy though to update the dimensions in the code to train for 256x256 or even higher if you want! The dataset is quite large, so it should work fine within reason.






