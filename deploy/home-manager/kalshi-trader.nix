# Optional Home Manager module for running kalshi-trader as a systemd user service.
# This file is NOT imported automatically. To use it, add to your Home Manager config:
#
#   imports = [ /home/<you>/Projects/kalshi-trader/deploy/home-manager/kalshi-trader.nix ];
#
# then `home-manager switch`. Remove the import to uninstall.
{ config, pkgs, ... }:
let
  projectDir = "${config.home.homeDirectory}/Projects/kalshi-trader";
in
{
  systemd.user.services.kalshi-trader = {
    Unit = {
      Description = "Kalshi tennis research engine (paper trading only, no real orders)";
      After = [ "network-online.target" ];
      Wants = [ "network-online.target" ];
    };
    Service = {
      Type = "simple";
      WorkingDirectory = projectDir;
      Environment = [ "PATH=${pkgs.nix}/bin:${pkgs.git}/bin:${pkgs.bash}/bin:${pkgs.coreutils}/bin" ];
      ExecStart = "${projectDir}/scripts/run-service.sh run";
      Restart = "on-failure";
      RestartSec = 15;
      KillSignal = "SIGTERM";
      TimeoutStopSec = 45;
    };
    Install.WantedBy = [ "default.target" ];
  };
}
