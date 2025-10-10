#!/bin/sh
if [ -f ".env" ]; then
  cp .env .devcontainer/.env.active
else
  # create an empty file so docker won't fail
  : > .devcontainer/.env.active
fi