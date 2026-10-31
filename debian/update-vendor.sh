#!/bin/sh
# Regenerates the vendored Rust sources that the package build consumes:
#
#   debian/vendor/                      crates.io
#   debian/vendor-ffier-0.2.0rc1/       ffier at tag 0.2.0rc1
#   debian/vendor-ffier-v0.2.0-rc2/     ffier at tag v0.2.0-rc2
#
# The workspace depends on two different commits of the ffier repository
# (src/libkrun uses v0.2.0-rc2, init and bindings use 0.2.0rc1). Both commits
# declare version 0.2.0, and a vendor directory cannot hold two packages with
# the same name and version, so each commit gets its own directory source.
# debian/cargo-home/config.toml maps the three sources.
#
# Needs network access. The package build itself runs offline (CARGO_NET_OFFLINE
# is set in debian/rules); replaced git sources require a Cargo.lock for the
# precise revisions, which the tree carries.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK="${ROOT}/.tmp/vendor-gen"
FFIER_URL="https://github.com/mtjhrc/ffier.git"
FFIER_RC1_TAG="0.2.0rc1"
FFIER_RC2_TAG="v0.2.0-rc2"
RC1_DIR="${ROOT}/debian/vendor-ffier-${FFIER_RC1_TAG}"
RC2_DIR="${ROOT}/debian/vendor-ffier-${FFIER_RC2_TAG}"

rm -rf "$WORK" "$ROOT/debian/vendor" "$RC1_DIR" "$RC2_DIR"
mkdir -p "$WORK/cargo-home"

# Directory source for the older ffier commit. Every crate the workspace pulls
# from that commit is reachable from these three dependencies.
mkdir -p "$WORK/rc1/src"
: >"$WORK/rc1/src/lib.rs"
cat >"$WORK/rc1/Cargo.toml" <<EOF
[package]
name = "ffier-rc1-vendor"
version = "0.0.0"
edition = "2021"

[dependencies]
ffier = { git = "${FFIER_URL}", tag = "${FFIER_RC1_TAG}" }
ffier-builtins = { git = "${FFIER_URL}", tag = "${FFIER_RC1_TAG}" }

[build-dependencies]
ffier-gen-rust-client = { git = "${FFIER_URL}", tag = "${FFIER_RC1_TAG}" }

[workspace]
EOF

CARGO_HOME="$WORK/cargo-home" cargo vendor --versioned-dirs \
	--manifest-path "$WORK/rc1/Cargo.toml" "$WORK/out-rc1" >/dev/null

mkdir -p "$RC1_DIR"
for crate in "$WORK"/out-rc1/ffier*; do
	mv "$crate" "$RC1_DIR/"
done

# Directory source for everything else, taken from the workspace's own lock
# file. Vendor refuses to emit both ffier commits, so the 0.2.0rc1 dependencies
# are pointed at the checkout cargo fetched above: path packages are passed over
# by vendor, which leaves the v0.2.0-rc2 commit to be vendored normally.
RC1_SRC=$(find "$WORK/cargo-home/git/checkouts" -mindepth 2 -maxdepth 2 -path '*/ffier-*/*' \
	| grep -m1 '/9616083b' || true)
if [ -z "$RC1_SRC" ]; then
	echo "could not locate the ffier 0.2.0rc1 checkout" >&2
	exit 1
fi

mkdir -p "$WORK/tree"
git -C "$ROOT" archive HEAD | tar -x -C "$WORK/tree"

RC1_SRC="$RC1_SRC" perl -pi -e '
	s{(\bffier[a-z-]*)(\s*)= \{ git = "https://github\.com/mtjhrc/ffier\.git", tag = "0\.2\.0rc1"(, optional = true)? \}}
	 {$1$2= { path = "$ENV{RC1_SRC}/$1"$3 }}g;
' "$WORK"/tree/init/init-blob/Cargo.toml \
	"$WORK"/tree/bindings/libkrun-via-cdylib-weak/Cargo.toml \
	"$WORK"/tree/bindings/init-blob-via-cdylib/Cargo.toml \
	"$WORK"/tree/src/devices/Cargo.toml

if grep -r "tag = \"${FFIER_RC1_TAG}\"" "$WORK/tree" --include=Cargo.toml; then
	echo "unrewritten ffier ${FFIER_RC1_TAG} dependency above" >&2
	exit 1
fi

CARGO_HOME="$WORK/cargo-home" cargo vendor --versioned-dirs \
	--sync "$ROOT/init/init-binary/Cargo.toml" "$ROOT/debian/vendor" \
	--manifest-path "$WORK/tree/Cargo.toml" >/dev/null

mkdir -p "$RC2_DIR"
for crate in "$ROOT"/debian/vendor/ffier*; do
	mv "$crate" "$RC2_DIR/"
done

# The package has no use for compiled objects, and dpkg-source would drop them
# from the source package anyway: it excludes *.a, *.la, *.o and *.so from
# anything under debian/ and its ignore list cannot be shortened. The four
# crates that ship objects (winapi's MinGW import libraries and wit-bindgen's
# prebuilt C-ABI runtime) are reachable only for Windows and wasm targets, so
# pruning them keeps the vendored tree, the source package and the build graph
# in agreement. The checksum manifests then have to lose the entries for the
# files that are gone; dh-cargo's per-crate manifests omit files for the same
# reason.
find debian/vendor debian/vendor-ffier-* -type f \
	\( -name '*.a' -o -name '*.la' -o -name '*.o' -o -name '*.so' \) -delete

find debian/vendor debian/vendor-ffier-* -name .cargo-checksum.json | while read -r checksum; do
	perl -MJSON::PP -e '
		my ($path) = @ARGV;
		$path =~ m{^(.*)/\.cargo-checksum\.json$} or die "$path: unexpected name\n";
		my $dir = $1;
		open my $in, "<", $path or die "$path: $!\n";
		my $data = JSON::PP->new->decode(do { local $/; <$in> });
		close $in;
		my %kept;
		for my $file (keys %{ $data->{files} }) {
			$kept{$file} = $data->{files}{$file} if -e "$dir/$file";
		}
		$data->{files} = \%kept;
		open my $out, ">", $path or die "$path: $!\n";
		print $out JSON::PP->new->canonical->encode($data), "\n";
		close $out;
	' "$checksum"
done

# dpkg-source refuses binary files in debian/ unless they are listed here. It
# treats a file as binary when its first 4 KiB contain a NUL byte, which is true
# of some of the test fixtures and images that ship inside the vendored crates.
find debian/vendor debian/vendor-ffier-* -type f | sort | perl -e '
	while (<>) {
		chomp;
		open my $fh, "<", $_ or next;
		binmode $fh;
		my $read = read $fh, my $buf, 4096;
		close $fh;
		print "$_\n" if defined $read and index($buf, "\0") >= 0;
	}
' >debian/source/include-binaries

rm -rf "$WORK"
