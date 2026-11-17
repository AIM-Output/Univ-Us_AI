#!/usr/bin/env python3
"""분할 zip(.z01 .z02 … + .zip, Info-ZIP `zip -s` / 7-Zip 형식)을 7-Zip 없이 풀거나 하나의 zip 으로 합친다.

  python unsplit_zip.py extract  qwen3-4b-f9-merged-nf4.zip  <출력폴더>
  python unsplit_zip.py join     qwen3-4b-f9-merged-nf4.zip  <합친.zip>

.z01 … 파일은 .zip 과 같은 폴더에 있어야 한다. 표준 라이브러리만 쓴다(Python 3.8+).
분할 zip 은 GitHub Release(파일당 2GiB 제한) 에 올린 nf4 병합본 `qwen3-4b-f9-merged-nf4.zip` + `.z01` 이다. 7-Zip 이 있으면 .zip 을 그냥 열어도 된다.
"""
import os, re, struct, sys, zlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # `python -I` 는 PYTHONUTF8 을 무시한다

SIG_CD, SIG_LH, SIG_EOCD, SIG_Z64_LOC, SIG_Z64_EOCD = 0x02014B50, 0x04034B50, 0x06054B50, 0x07064B50, 0x06064B50


class Disks:
    """disk 0..N-1 를 이어 붙인 한 스트림처럼 읽는다 (.z01 = disk 0, … .zip = 마지막 disk)."""
    def __init__(self, last_zip: Path):
        parts = sorted(last_zip.parent.glob(last_zip.stem + ".z[0-9][0-9]"), key=lambda p: int(p.suffix[2:]))
        self.files = parts + [last_zip]
        self.sizes = [p.stat().st_size for p in self.files]
        self.base = [sum(self.sizes[:i]) for i in range(len(self.files))]
        self.total = sum(self.sizes)
        self.fh = [open(p, "rb") for p in self.files]

    def read(self, off: int, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            i = max(k for k, b in enumerate(self.base) if b <= off)
            f = self.fh[i]; f.seek(off - self.base[i])
            chunk = f.read(min(n, self.base[i] + self.sizes[i] - off))
            if not chunk: raise EOFError(f"offset {off}")
            out += chunk; off += len(chunk); n -= len(chunk)
        return bytes(out)

    def abs_off(self, disk: int, rel: int) -> int:
        return self.base[disk] + rel


def parse_central_dir(d: Disks):
    tail = d.read(max(0, d.total - 66000), min(66000, d.total))
    i = tail.rfind(struct.pack("<I", SIG_EOCD))
    if i < 0: raise SystemExit("EOCD 를 찾지 못함 — zip 이 아니거나 마지막 조각이 아니다")
    eocd = tail[i:i + 22]
    _, this_disk, cd_disk, n_this, n_total, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", eocd)
    if 0xFFFF in (cd_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_off):
        loc = tail[i - 20:i]
        if struct.unpack("<I", loc[:4])[0] != SIG_Z64_LOC: raise SystemExit("zip64 locator 없음")
        _, z64_disk, z64_off, _ = struct.unpack("<IIQI", loc)
        z = d.read(d.abs_off(z64_disk, z64_off), 56)
        _, _, _, _, _, cd_disk, _, n_total, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", z)
    pos = d.abs_off(cd_disk, cd_off); end = pos + cd_size
    entries = []
    while pos < end:
        h = d.read(pos, 46)
        if struct.unpack("<I", h[:4])[0] != SIG_CD: raise SystemExit(f"central dir 깨짐 @ {pos}")
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _, lh_off) = struct.unpack("<IHHHHHHIIIHHHHHII", h)
        name = d.read(pos + 46, nlen).decode("utf-8" if flags & 0x800 else "cp437")
        extra = d.read(pos + 46 + nlen, xlen)
        j = 0
        while j + 4 <= len(extra):
            eid, elen = struct.unpack("<HH", extra[j:j + 4]); body = extra[j + 4:j + 4 + elen]
            if eid == 0x0001:
                k = 0
                for field in ("usize", "csize", "lh_off", "disk"):
                    cur = {"usize": usize, "csize": csize, "lh_off": lh_off, "disk": disk}[field]
                    sentinel = 0xFFFF if field == "disk" else 0xFFFFFFFF
                    if cur == sentinel:
                        width = 4 if field == "disk" else 8
                        val = struct.unpack("<I" if width == 4 else "<Q", body[k:k + width])[0]; k += width
                        if field == "usize": usize = val
                        elif field == "csize": csize = val
                        elif field == "lh_off": lh_off = val
                        else: disk = val
            j += 4 + elen
        entries.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, abs_lh=d.abs_off(disk, lh_off), flags=flags))
        pos += 46 + nlen + xlen + clen
    return entries


def data_start(d: Disks, e: dict) -> int:
    h = d.read(e["abs_lh"], 30)
    if struct.unpack("<I", h[:4])[0] != SIG_LH: raise SystemExit(f"local header 깨짐: {e['name']}")
    nlen, xlen = struct.unpack("<HH", h[26:30])
    return e["abs_lh"] + 30 + nlen + xlen


def extract(last_zip: Path, out_dir: Path):
    d = Disks(last_zip); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"조각 {len(d.files)}개 · {d.total/2**30:.2f} GiB → {out_dir}")
    for e in parse_central_dir(d):
        target = out_dir / e["name"]
        if e["name"].endswith("/"): target.mkdir(parents=True, exist_ok=True); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        off, left, crc = data_start(d, e), e["csize"], 0
        dec = zlib.decompressobj(-15) if e["method"] == 8 else None
        if e["method"] not in (0, 8): raise SystemExit(f"압축 방식 {e['method']} 미지원: {e['name']}")
        with open(target, "wb") as w:
            while left > 0:
                chunk = d.read(off, min(left, 8 << 20)); off += len(chunk); left -= len(chunk)
                data = dec.decompress(chunk) if dec else chunk
                crc = zlib.crc32(data, crc); w.write(data)
            if dec:
                data = dec.flush(); crc = zlib.crc32(data, crc); w.write(data)
        ok = (crc & 0xFFFFFFFF) == e["crc"]
        print(f"  {'OK ' if ok else 'CRC 불일치'} {e['name']}  {e['usize']:,} B")
        if not ok: raise SystemExit(f"CRC 불일치: {e['name']}")
    print("완료")


def join(last_zip: Path, out_zip: Path):
    """조각을 하나의 보통 zip 으로 — 중앙 디렉터리의 disk 번호·오프셋을 이어 붙인 스트림 기준으로 고쳐 쓴다 (zip64 아닌 경우)."""
    d = Disks(last_zip)
    if d.total >= 0xFFFFFFFF: raise SystemExit("4GiB 이상은 join 대신 extract 를 쓰세요")
    entries = parse_central_dir(d)
    with open(out_zip, "wb") as w:
        start = 4 if d.read(0, 4) == struct.pack("<I", 0x08074B50) else 0   # 분할 서명 PK\x07\x08 는 뗀다
        first_cd = min(e["abs_lh"] for e in entries) if entries else d.total
        # 데이터 구간(로컬 헤더 ~ 중앙 디렉터리 직전)을 그대로 복사
        tail = d.read(max(0, d.total - 66000), min(66000, d.total)); i = tail.rfind(struct.pack("<I", SIG_EOCD))
        _, _, cd_disk, _, n_total, cd_size, cd_off, clen = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        cd_abs = d.abs_off(cd_disk, cd_off)
        pos = start
        while pos < cd_abs:
            chunk = d.read(pos, min(cd_abs - pos, 8 << 20)); w.write(chunk); pos += len(chunk)
        shift = start
        cd = bytearray(d.read(cd_abs, cd_size)); p = 0
        while p < len(cd):
            nlen, xlen, clen2 = struct.unpack("<HHH", cd[p + 28:p + 34]); disk, = struct.unpack("<H", cd[p + 34:p + 36]); lh, = struct.unpack("<I", cd[p + 42:p + 46])
            if lh != 0xFFFFFFFF:
                cd[p + 42:p + 46] = struct.pack("<I", d.abs_off(disk, lh) - shift)
            cd[p + 34:p + 36] = struct.pack("<H", 0)
            p += 46 + nlen + xlen + clen2
        w.write(cd)
        w.write(struct.pack("<IHHHHIIH", SIG_EOCD, 0, 0, n_total, n_total, cd_size, cd_abs - shift, 0))
    print(f"→ {out_zip} ({out_zip.stat().st_size/2**30:.2f} GiB)")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("extract", "join"): print(__doc__); sys.exit(2)
    (extract if sys.argv[1] == "extract" else join)(Path(sys.argv[2]), Path(sys.argv[3]))
