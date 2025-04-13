Music Genre Classification
This project classifies music tracks into genres using machine learning. It extracts audio features like MFCCs and uses a Random Forest classifier for prediction.

Requirements:
Install required libraries:

bash
Copy
Edit
pip install librosa scikit-learn numpy pandas matplotlib seaborn
Dataset:
Organize your music files into genre folders like this:

markdown
Copy
Edit
dataset/
    rock/
        song1.mp3
        song2.mp3
    pop/
        song1.mp3
        song2.mp3
Setup and Usage:
Download or clone the repo.

Update the dataset path in the code.

Run the code:

bash
Copy
Edit
python music_genre_classification.py
It will print the accuracy and show a confusion matrix.

License:
MIT License.
