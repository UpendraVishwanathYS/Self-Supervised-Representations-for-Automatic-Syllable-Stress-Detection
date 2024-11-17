# Self-Supervised-Representations-for-Automatic-Syllable-Stress-Detection
Uncovering Lexical Stress Patterns: A Layered Analysis of Self-Supervised Representations for Automatic Syllable Stress Detection

# Installing Dependencies
To install dependencies, create a conda or virtual environment with Python 3 and then run ```pip install -r requirements.txt```

# Training Supervised Models for Syllable-Stress-Detection
To train the Deep Neural Network (DNN) on w2v2-base-960h (L1) features simply run ```python3 main.py```. You can also add custom parameters in the command line:```python train_model.py --layer_number --language --w2v2_model_name --classification_model --wav_files_path_zip --path_to_database --checkpoint_path --epochs 5 --batch_size 32```

The default values are specified below:
```
  parser.add_argument('--layer_number', type=int, default=1, help='Layer number to process')
  parser.add_argument('--language', type=str, default='GER', help='Language code (e.g., GER)')
  parser.add_argument('--w2v2_model_name', type=str, default='wav2vec2-base-960h', help='Name of the model (e.g., wav2vec2-base-960h)')
  parser.add_argument('--classification_model', type=str, default='DNN', help='Classification model (e.g., DNN)')
  parser.add_argument('--wav_files_path_zip', type=str, default='./wav_final.zip', help='Path to the zip file containing audio files')
  parser.add_argument('--path_to_database', type=str, default='./database.csv', help='Path to the database CSV file')
  parser.add_argument('--checkpoint_path', type=str, default='./', help='Path to the checkpoint directory')
  parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
```
# Extract syllable level features using W2V2 models:
To analyze and extract syllable level w2v2 feature from an audio file utilize the ```wav2vec2_audio_feature_extraction.py```. The default model in the script is ```facebook\wav2vec-base-960h```. The inputs include (1) path to the audio file (or) time series of audio file (2) type of w2v2 model (3) start and end time of the syllable
