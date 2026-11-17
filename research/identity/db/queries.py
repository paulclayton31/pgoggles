"""Interface for reading/writing person records keyed by facial embedding."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from .tables import Person


def add_person(session: Session, name: str, embedding: list[float]) -> Person:
    person = Person(name=name, embedding=embedding)
    session.add(person)
    session.commit()
    session.refresh(person)
    return person


def find_nearest(session: Session, embedding: list[float], limit: int = 1) -> list[Person]:
    stmt = select(Person).order_by(Person.embedding.l2_distance(embedding)).limit(limit)
    return list(session.scalars(stmt))


def get_person_by_id(session: Session, person_id: int) -> Person | None:
    return session.get(Person, person_id)


def delete_person(session: Session, person_id: int) -> bool:
    person = session.get(Person, person_id)
    if person is None:
        return False
    session.delete(person)
    session.commit()
    return True


def delete_all_persons(session: Session) -> None:
    session.query(Person).delete()
    session.commit()
