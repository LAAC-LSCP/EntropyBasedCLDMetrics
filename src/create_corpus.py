from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from utterances_cleaner_thomas import UtterancesCleaner
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pylangacq
from datetime import datetime
import os

CLEANER = UtterancesCleaner("extra/markers.json")

def get_data(child_corpus, annotation_campaign):
    
    children = pd.read_csv(f"{child_corpus}/metadata/children.csv")
    recordings = pd.read_csv(f"{child_corpus}/metadata/recordings.csv")
    
    project = ChildProject(child_corpus)
    am = AnnotationManager(project)
    am.read()
    
    annotation = am.annotations[am.annotations['set'].str.startswith(annotation_campaign)]
    annotation_set = annotation['set'].unique()[0]
    
    annotation_files = annotation['annotation_filename'].unique()
    
    for annotation_file in tqdm(annotation_files):
        
        path = f"{child_corpus}/annotations/{annotation_set}/converted/{annotation_file}"
        annots = pd.read_csv(path)
        
        needed_columns = zip(annots["transcription"], annots["segment_onset"],
                             annots["segment_offset"], annots["speaker_role"])
        
        recording_filename = annotation[annotation['annotation_filename'] == annotation_file]['recording_filename'].values[0]
        child_id = recordings[recordings['recording_filename'] == recording_filename]['child_id'].values[0]
        recording_date = datetime.strptime(recordings[recordings['recording_filename'] == recording_filename]['date_iso'].values[0], "%Y-%m-%d")
        DOB = datetime.strptime(children[children['child_id'] == child_id]['child_dob'].values[0], "%Y-%m-%d")
        age = (recording_date - DOB).days//30
        
        data = {"age" : age, "data": defaultdict(list), "filename": recording_filename.split('.')[0]}
        for utterance_raw, onset, offset, speaker_role in needed_columns:
            utterance = pylangacq.chat._clean_utterance(utterance_raw)
            cleaned = CLEANER.clean(utterance)
            data["data"][speaker_role].append((utterance_raw, cleaned, onset, offset))
        yield data

def write_utterances(utterances, output_file):
    """Writes utterances in a given file."""
    with open(output_file, "w") as utterance_file:
        for utterance in utterances:
            utterance_file.write(f"{utterance}\n")

def make_folder(childes_corpus, annotation_campaign, output_folder):
    """
    Creates the folders for the different utterances types:\
    orthographic, cleaned, timemarks.
    """
    child_name = childes_corpus.split('/')[-1]
    print(child_name)
    allowed_speakers = {"Mother", "Target_Child"}
    for data in get_data(childes_corpus, annotation_campaign):
        age_orthographic_folder = f"{output_folder}/orthographic/{child_name}/{data['filename']}"
        os.makedirs(age_orthographic_folder, exist_ok=True)
        age_cleaned_folder = f"{output_folder}/cleaned/{child_name}/{data['filename']}"
        os.makedirs(age_cleaned_folder, exist_ok=True)
        age_timemarks_folder = f"{output_folder}/timemarks/{child_name}/{data['filename']}"
        os.makedirs(age_timemarks_folder, exist_ok=True)

        for speaker_role in data["data"]:
            speaker_data = list(zip(*data["data"][speaker_role]))
            utterances, cleaneds, onsets, offsets = speaker_data
            timemarks = [f"{onset}\t{offset}" for onset, offset in zip(onsets, offsets)]
            if speaker_role not in allowed_speakers:
                continue
            assert len(utterances) == len(cleaneds) == len(timemarks), "Mismatch in the data"

            utterance_orthographic_output = f"{age_orthographic_folder}/{speaker_role}.orthographic"
            utterance_cleaned_output = f"{age_cleaned_folder}/{speaker_role}.cleaned"
            utterance_timemarks_output = f"{age_timemarks_folder}/{speaker_role}.timemarks"
        
            write_utterances(utterances, utterance_orthographic_output)
            write_utterances(cleaneds, utterance_cleaned_output)
            write_utterances(timemarks, utterance_timemarks_output)
        
        with open(age_orthographic_folder + "/months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_orthographic_folder + "/filename.txt", "w") as filename_file:
            filename_file.write(str(data["filename"]))

        with open(age_cleaned_folder + "/months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_cleaned_folder + "/filename.txt", "w") as filename_file:
            filename_file.write(str(data["filename"]))

        with open(age_timemarks_folder + "/months.txt", "w") as months_file:
            months_file.write(str(data["age"]))
        with open(age_timemarks_folder + "/filename.txt", "w") as filename_file:
            filename_file.write(str(data["filename"]))
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--childes_corpus",
                        help="Folder containing childes corpus.",
                        required=True)
    parser.add_argument("-a", "--annotation_campaign",
                        help="Annotation campaign of the childes corpus.",
                        required=True)
    parser.add_argument("-o", "--output_folder",
                        help="Where the folder will be stored",
                        required=True)

    args = parser.parse_args()
    make_folder(args.childes_corpus, args.annotation_campaign, args.output_folder)
    
    