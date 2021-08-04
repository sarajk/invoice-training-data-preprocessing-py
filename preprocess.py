import csv
import sys
import json
from pathlib import Path
import math
import datasaver

def find_feature_files(dir_path):

    dir = Path(dir_path)

    # Check if the directory exists and it's actually a directory
    if not dir.exists() or not dir.is_dir():
        raise RuntimeError('Did you forget to specify the invoice-feature-extraction directory?')

    # Fetch all files recursively inside the directory that match features.json
    files = list(dir.rglob("features.json"))

    # Convert from Path object to an actual absolute path as a string
    return [str(f) for f in files]


def read_features(feature_files):

    features_list = []

    for feature_file in feature_files:
        # Read each feature file
        with open(feature_file, 'r') as file:
            # Read the json file as a dictionary/list
            features = json.load(file)
            for feature in features:
                # Set the origin for the feature because it's going to be used
                # to know with what to match it later
                feature['origin'] = feature_file
                # Add each feature inside the file into a combining list
                features_list.append(feature)
    return features_list

def convert(features_list):

    training_data_list = []

    for feature in features_list:

        if 'boundingBox' not in feature:
            continue
        
        boundingBox = feature['boundingBox']

        for testFeature in features_list:

            # If we're referencing another feature with the same ID
            # i.e: The same features from different files
            if feature != testFeature and feature['id'] == testFeature['id']:
                continue

            # We only map incorrect combinations of features if they're from the same file
            # This is done just because images might vary in size, position, and other variables
            if feature['origin'] != testFeature['origin']:
                continue
            
            keyX = boundingBox['x']
            keyY = boundingBox['y']
            keyWidth = boundingBox['width']
            keyHeight = boundingBox['height']

            for valueBoundingBox in testFeature['valueBoundingBoxes']:

                valueX = valueBoundingBox['x']
                valueY = valueBoundingBox['y']
                valueWidth = valueBoundingBox['width']
                valueHeight = valueBoundingBox['height']

                training_data = {
                    'id': feature['id'],
                    # If it's the feature itself, then label it correctly (Date, Cash collected...) otherwise label it as Incorrect
                    # This will be needed for after training
                    'label': feature['key'] if (feature == testFeature) else 'Incorrect',
                    # The distance is computed using Euclidean distance
                    'distance': math.sqrt(math.pow(valueX - keyX, 2) + math.pow(valueY - keyY, 2)),

                    'width': valueWidth
                }

                if ((valueX + valueWidth / 2) < keyX and (valueY + valueHeight) < keyY) :
                    training_data['position'] = 1 # ABOVE_LEFT
                elif (valueX > (keyX + keyWidth / 2) and (valueY + valueHeight) < keyY) :
                    training_data['position'] = 2 # ABOVE_RIGHT
                elif ((valueX + valueWidth / 2) < keyX and (keyY + keyHeight) < valueY) :
                    training_data['position'] = 6 # BELOW_LEFT
                elif (valueX > (keyX + keyWidth / 2) and (keyY + keyHeight) < valueY) :
                    training_data['position'] = 7 # BELOW_RIGHT
                elif ((valueX + valueWidth) < keyX) :
                    training_data['position'] = 3 # LEFT
                elif ((valueY + valueHeight) < keyY) :
                    training_data['position'] = 0 # ABOVE
                elif (valueX > (keyX + keyWidth)) :
                    training_data['position'] = 4 # RIGHT
                else :
                    training_data['position'] = 5 # BELOW
                
                training_data_list.append(training_data)
				
    return training_data_list

def normalize(training_data_list):

    normalized_training_list = []

    maxWidth = max([td['width'] for td in training_data_list])
    print('maxWidth =',maxWidth)
    minWidth = min([td['width'] for td in training_data_list])
    print('minWidth =',minWidth)

    maxDistance = max([td['distance'] for td in training_data_list])
    print('maxDistance =',maxDistance)
    minDistance = min([td['distance'] for td in training_data_list])
    print('minDistance =',minDistance)

    # Save the normalized params
    with open('normalized_params.json', 'w') as f:
        json.dump({
            'maxWidth':maxWidth,
            'minWidth':minWidth,
            'maxDistance':maxDistance,
            'minDistance':minDistance
        }, f)

    for training_data in training_data_list:

        normalized_training_data = {
            'id':training_data['id'],
            'label':training_data['label'],
            'width': (training_data['width'] - minWidth) / (maxWidth - minWidth),
            'distance': (training_data['distance'] - minDistance) / (maxDistance - minDistance),
            'position': training_data['position']
        }

        normalized_training_list.append(normalized_training_data)
    return normalized_training_list



# NOTE: DON'T FORGET TO SPECIFY THE DIRECTORY CONTAINING THE features.json FILES
features_dir_path = 'examples/input'

if len(features_dir_path.strip()) == 0:
    print('*** Did you forget to specify the features_dir_path directory? ***')
    exit(1)

feature_files = find_feature_files(features_dir_path)

print('Found', len(feature_files), 'features.json file(s).')

training_data_list = convert(read_features(feature_files))

print('Raw features have been converted to training data.')

datasaver.write_csv(training_data_list, 'output-with-wrong.csv')

print('Training data saved onto output-with-wrong.csv')

normalized_training_data_list = normalize(training_data_list)

print('Normalized training data saved onto output-with-wrong.csv')

datasaver.write_csv(normalized_training_data_list, 'normalized-with-wrong.csv')