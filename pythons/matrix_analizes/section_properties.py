class SectionProperties:
    def __init__(self, id, E, A, I):
        self.id = id
        self.E = E
        self.A = A
        self.I = I

    def __repr__(self):
        return f"SectionProperties(id={self.id}, E={self.E}, A={self.A})"
