import csv


def csv_to_tsv(csv_file, tsv_file):
    with open(csv_file, 'r') as csv_in, open(tsv_file, 'w') as tsv_out:
        csv_reader = csv.reader(csv_in)
        tsv_writer = csv.writer(tsv_out, delimiter='\t')
        for row in csv_reader:
            tsv_writer.writerow(row)


def run(dataset):
    # TRAIN
    filename_train_csv = dataset + "_TRAIN.csv"
    filename_train_tsv = dataset + "_TRAIN.tsv"
    csv_to_tsv(filename_train_csv, filename_train_tsv)

    # TEST
    # TRAIN
    filename_test_csv = dataset + "_TEST.csv"
    filename_test_tsv = dataset + "_TEST.tsv"
    csv_to_tsv(filename_test_csv, filename_test_tsv)


if __name__ == '__main__':
    run("Charging")
