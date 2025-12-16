import hashlib
import json


def hash_hypergraph(hypergraph):
    """
    Generates a SHA-256 hash of a hypergraph based on its exposed attributes.

    Parameters
    ----------
    hypergraph : object
        The hypergraph instance to hash. Should implement `expose_attributes_for_hashing`.

    Returns
    -------
    str
        The SHA-256 hash hex digest of the hypergraph.
    """

    def serialize(obj):
        """
        Recursively serialize the hypergraph attributes into a JSON-compatible structure,
        ensuring that dictionaries are sorted by key and lists are sorted if applicable.
        """
        if isinstance(obj, dict):
            return {k: serialize(obj[k]) for k in sorted(obj)}
        elif isinstance(obj, list):
            return [serialize(item) for item in obj]
        else:
            return obj

    exposed_attrs = hypergraph.expose_attributes_for_hashing()

    serialized_hg = serialize(exposed_attrs)

    json_str = json.dumps(serialized_hg, sort_keys=True)

    hash_digest = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    return hash_digest
