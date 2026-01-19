{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            python.pkgs.pip
            python.pkgs.virtualenv
            pkgs.postgresql_16
            pkgs.git
          ];

          shellHook = ''
            export PGDATA="$PWD/.postgres"
            export PGHOST="$PGDATA"

            if [ ! -d "$PGDATA" ]; then
              initdb -D "$PGDATA"
              echo "unix_socket_directories = '$PGDATA'" >> "$PGDATA/postgresql.conf"
              echo "listen_addresses = ''" >> "$PGDATA/postgresql.conf"
            fi

            [ ! -d "venv" ] && python -m venv venv
            source venv/bin/activate
            [ ! -f ".env" ] && cp .env.example .env 2>/dev/null || true

            echo "✓ Nix dev shell ready"
            echo "Run: pip install -e .[dev]"
          '';
        };
      });
}
