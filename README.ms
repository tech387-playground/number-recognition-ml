# ML Demo: Numbers recognition

## Note
In order to have a good speed to load the training data in your project you will have to use 
`docker-sync`.
The issue which happens is that your source code is mounted from your osx host’s filesytem to your Docker filesystem. 
For them to communicate together, every call to your mounted volume uses the osxfs shared file system solution.

We will undergoo the following steps to resolve the issue:
- turned into a FUSE message in the Linux VFS
- proxied over a virtio socket by transfused
- forwarded onto a UNIX domain socket by HyperKit
deserialized, dispatched and executed as a macOS system call by osxfs

Link to install `docker-sync`: https://docker-sync.readthedocs.io/en/latest/getting-started/installation.html

## Configure docker-sync

Let’s say you have a Dockerfile and a docker-compose.yaml. A minimal docker-compose.yaml would look something like that:
`
version: "3.1"
services:
  web:
    build: .
    restart: always
    volumes:
      - api-sync:/usr/src/app

volumes:
  api-sync:
    external: true
`

Create a docker-sync.yaml file and past in this configuration:
`
version: "2"
syncs:
  api-sync:
    sync_strategy: 'native_osx'
    src: '.' #path to the volume you want to synchronise
    host_disk_mount_mode: 'cached'
`