from pathlib import Path

from idtrackerai import ListOfBlobs, ListOfFragments, Session
from idtrackerai.utils import wrap_entrypoint


def populate_groundtruths(blobs: ListOfBlobs, fragments: ListOfFragments):
    identity_sets: list[set[int]] = [set() for _ in fragments]
    for blob in blobs.all_blobs:
        identities = list(blob.final_identities)
        identity = 0 if len(identities) != 1 or identities[0] is None else identities[0]
        identity_sets[blob.fragment_identifier].add(identity)

    for frag, identity_set in zip(fragments, identity_sets):
        if frag.is_a_crossing:
            frag.groundtruth_identity = 0
            continue

        if len(identity_set) != 1:
            frag.groundtruth_identity = 0
            continue

        frag.groundtruth_identity = identity_set.pop()


@wrap_entrypoint
def main():
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("session_folders", nargs="+", type=Path)
    args = parser.parse_args()

    for session_path in args.session_folders:
        session = Session.load(session_path)
        if not session.timers["Tracking session"].finished:
            logging.warning(f"{session} not finished, skipping groundtruth")
            continue
        if not (session_path / "trajectories" / "validated.npy").is_file():
            logging.warning(f"{session} not validated, skipping groundtruth")
            continue
        blobs = ListOfBlobs.load(session.blobs_path)
        fragments = ListOfFragments.load(session.fragments_path, False)
        populate_groundtruths(blobs, fragments)
        fragments.save(session.fragments_path)


if __name__ == "__main__":
    main()
