# Natural Language Processing of Lyrics Classification Project

## Global Description
<div style="text-align: justify;">
This project involves the development and evaluation of machine and deep learning models for classifying song lyrics according different artists names. Various models, including baseline and enhanced versions, have been implemented and evaluated to achieve optimal classification performance.
The objective was to know if simple Learning Technique could result in generating a model that could differentiate the artists by their style of writing.
The use of Natural Language Processing technique was then mandatory to reach this goal.
</div>

## Dataset
<div style="text-align: justify;">
The dataset used for this project is available at <a href='https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres/'>Kaggle</a>. It contains song lyrics from artists such as Drake, Kanye West, 50 Cent, Taylor Swift, Celine Dion, and Rihanna.
</div>

## Model Performances

| Model Description                                   | Accuracy | Precision | Recall | 
|-----------------------------------------------------|----------|-----------|--------|
| Baseline Model                                      | 0.79     | 0.80      | 0.79   | 
| Improved Baseline Model - OverSampling and HyperParams | 0.83  | 0.84      | 0.83   |
| LSTM Layer Implementation - No Other Modification   | 0.59     | 0.60      | 0.59   |
| LSTM Layer Implementation - Early Stopping + Oversampling + RNN Layers | 0.65 | 0.64      | 0.65   |

# Installation and Usage

To run this project, follow these steps:

1. Clone the repository : `git clone https://github.com/your-username/NLP_PROJECT.git`
2. Navigate to the project directory : `cd NLP_PROJECT`
3. Install dependencies : `pip install -r requirements.txt`
4. Download the dataset as lyrics-data.csv filename (link in references). Put it in the data folder.
5. Run the notebooks : Just execute each cell.

# References

- Kaggle Dataset : [Scrapped Lyrics from 6 Genres](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres/)
- StackOverFlow : [For a lot of different topics](https://stackoverflow.com/)
- ChatGPT : [For the comments and some part of codes (charts for example)](https://chat.openai.com/)
- Previous Labs Made Together