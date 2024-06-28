class NodeNotFoundError(Exception):
    def __init__(self, node_key):
        super().__init__(f"Node with key '{node_key}' not found inside workflow")


class TooManyCharactersError(Exception):
    def __init__(self, n_characters, max_characters):
        super().__init__(
            f"Too many characters specified, you specified {n_characters}, use at most {max_characters} characters"
        )


class CharacterIdNotInRangeError(Exception):
    def __init__(self, char_id, max_character_id):
        super().__init__(f"Character ID {char_id} is not in range 0-{max_character_id}")
