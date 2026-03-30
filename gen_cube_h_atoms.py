
class GenCubeHAtoms:

    def __init__(self, natoms, distance, charge=0, multiplicity=1):
        self.natoms = natoms
        self.dis = distance
        self.charge = charge
        self.mult = multiplicity

    def generate_particles(self, origin=(0, 0, 0)):
        x0, y0, z0 = origin
        particles = []

        for i in range(self.natoms):
            for j in range(self.natoms):
                for k in range(self.natoms):
                    x = x0 + i * self.dis
                    y = y0 + j * self.dis
                    z = z0 + k * self.dis
                    particles.append((x, y, z))

        return particles

    def to_xyz(self, filename=None):
        if filename is None:
            filename = f"cube_h_{self.natoms}_d_{self.dis:.2f}_bohr.xyz"
        particles = self.generate_particles()
        n_total = len(particles)

        with open(filename, "w") as f:
            # XYZ header
            f.write("units bohr\n")
            f.write(f"{self.charge} {self.mult}\n")

            # atom lines
            for (x, y, z) in particles:
                f.write(f"H {x:.4f} {y:.4f} {z:.4f}\n")

        print(f"File '{filename}' written with {n_total} atoms.")    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gen_cube_h_atoms.py <natoms> <distance>")
        sys.exit(1)

    try:
        natoms = int(sys.argv[1])
        distance = float(sys.argv[2])
    except ValueError:
        print("Error: natoms must be int, distance must be float")
        sys.exit(1)

    cal = GenCubeHAtoms(natoms, distance)
    cal.to_xyz()