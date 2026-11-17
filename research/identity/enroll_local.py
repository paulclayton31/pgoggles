import cv2

from research.identity.db.database import get_session
from research.identity.db.queries import add_person
from research.identity.embeddings.face_embedder import FaceEmbedder


def main():
    name = input("Enter person's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    print(f"Enrolling: {name}")
    print("Press SPACE to capture and enroll.")
    print("Press Q to quit.")

    embedder = FaceEmbedder()
    session = get_session()

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                continue

            cv2.imshow("PGoggles Face Enrollment", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Enrollment cancelled.")
                break

            if key == 32:  # SPACE
                print("Detecting face...")

                embedding = embedder.embed(frame)

                if embedding is None:
                    print("No face detected. Try again.")
                    continue

                person = add_person(
                    session,
                    name=name,
                    embedding=embedding,
                )

                print(
                    f"Enrolled successfully: "
                    f"id={person.id}, name={person.name}"
                )

                break

    finally:
        session.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()