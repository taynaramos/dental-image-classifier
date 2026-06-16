"""Main entry point for the project."""

from src.data import loader
from src.preprocessing import clean_data, split_data


def main():
    data = loader("")  # add path
    cleaned_data = clean_data(data)  # fix returned value
    split_data(cleaned_data)


if __name__ == "__main__":
    main()
