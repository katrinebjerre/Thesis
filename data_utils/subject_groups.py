# data_utils/subject_groups.py

# Subjects reserved for person-specific use and testing (excluded from general LOTO model)
PERSON_SPECIFIC_SUBJECTS = [
    "002",
    "007",
    "023",
    "034",
    "037",
]


def get_person_specific_subjects():
    """Returns the list of reserved subject IDs."""
    return PERSON_SPECIFIC_SUBJECTS