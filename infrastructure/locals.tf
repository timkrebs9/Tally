locals {
  tags = {
    Owner       = "timkrebs"
    Project     = "tally"
    Environment = "${var.environment}"
    Toolkit     = "terraform"
    Name        = "${var.prefix}"
  }
}