from typing import List

MIN_LENGTH = 7


def count_names_longer_than(prenoms: List[str], min_length: int = MIN_LENGTH) -> int:
    """
    Count how many names are strictly longer than a given length.
    """
    return sum(1 for prenom in prenoms if len(prenom) > min_length)


if __name__ == "__main__":
    prenoms = [
        "Guillaume",
        "Gilles",
        "Juliette",
        "Antoine",
        "François",
        "Cassandre",
    ]
    result = count_names_longer_than(prenoms)
    print(f"Nombre de prénoms avec plus de {MIN_LENGTH} lettres : {result}")
