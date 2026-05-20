"""Lightweight PDB parsing used by DINA2 feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


@dataclass
class Atom:
    name: str
    x: float
    y: float
    z: float
    bfactor: float
    element: str


@dataclass
class Residue:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atoms: dict[str, Atom] = field(default_factory=dict)

    @property
    def aa(self) -> str:
        return AA3_TO_1.get(self.resname.upper(), "X")

    @property
    def residue_id(self) -> str:
        code = self.icode.strip()
        return f"{self.chain_id}:{self.resseq}{code}"

    @property
    def sidechain_atoms(self) -> list[Atom]:
        return [atom for name, atom in self.atoms.items() if name not in BACKBONE_ATOMS and atom.element != "H"]


def parse_pdb(path: str) -> list[Residue]:
    """Parse ATOM records from a PDB file into residues."""

    residues: list[Residue] = []
    by_key: dict[tuple[str, int, str], Residue] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            chain_id = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
            except ValueError:
                continue
            icode = line[26].strip()
            element = (line[76:78].strip() or atom_name[0]).upper()
            key = (chain_id, resseq, icode)
            residue = by_key.get(key)
            if residue is None:
                residue = Residue(chain_id=chain_id, resseq=resseq, icode=icode, resname=resname)
                by_key[key] = residue
                residues.append(residue)
            residue.atoms[atom_name] = Atom(atom_name, x, y, z, bfactor, element)
    return residues


def residues_to_sequence(residues: list[Residue]) -> str:
    return "".join(res.aa for res in residues)
