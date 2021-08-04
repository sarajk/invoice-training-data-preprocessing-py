import csv

def write_csv(training_data_list, file_name):

    if len(training_data_list) == 0:
        raise RuntimeError('Cannot save file ' + file_name + ' because the training_data_list is empty!')

    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = list(training_data_list[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')

        writer.writeheader()

        for training_data in training_data_list:
            writer.writerow(training_data)