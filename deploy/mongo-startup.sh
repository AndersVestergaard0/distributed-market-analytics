#! /bin/bash
# Startup script for the MongoDB GCE VM (see GCP_DEPLOYMENT.md Part 2).
#
# Installs Docker (via get.docker.com - Debian 12's default repos don't
# carry docker-compose-plugin, which silently broke earlier versions of this
# script), then brings up mongo + mongo-express with auth enabled:
#   - "writer"        readWrite on market_analytics - used by the Dataproc
#                      pipeline jobs (MONGO_URI, Part 3) to write results.
#   - "public_reader"  read-only on market_analytics - the only credential
#                      mongo-express itself holds, baked into its own
#                      container config and never sent to the browser. This
#                      is what lets the mongo-express port be opened to the
#                      public internet (Part 2's firewall rule) without
#                      letting a visitor modify the results: any write
#                      attempted through the UI is rejected by MongoDB
#                      itself, regardless of what the UI renders.
#
# Runs idempotently on every boot: MONGO_INITDB_ROOT_USERNAME/PASSWORD only
# take effect on first init of an empty /data/db (persisted via the
# mongo_data volume), and the createUser calls below are safe to repeat
# (they no-op with an error on an existing user, which is why they're
# suffixed with "|| true").

set -e

curl -fsSL https://get.docker.com | sh

meta() {
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
ROOT_PW="$(meta MONGO_ROOT_PASSWORD)"
WRITER_PW="$(meta MONGO_WRITER_PASSWORD)"
READER_PW="$(meta MONGO_READER_PASSWORD)"

mkdir -p /opt/mongo && cd /opt/mongo
cat > docker-compose.yml <<COMPOSE
services:
  mongo:
    image: mongo
    container_name: mongo
    restart: always
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: $ROOT_PW
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"
  mongo_admin:
    image: mongo-express
    container_name: mongoadmin
    restart: always
    depends_on:
      - mongo
    environment:
      ME_CONFIG_MONGODB_URL: mongodb://public_reader:$READER_PW@mongo:27017/market_analytics?authSource=market_analytics
      ME_CONFIG_BASICAUTH: "false"
    ports:
      - "8081:8081"
volumes:
  mongo_data:
COMPOSE

docker compose up -d

# wait for mongo to accept connections before provisioning app users
for i in $(seq 1 30); do
  docker exec mongo mongosh -u admin -p "$ROOT_PW" --authenticationDatabase admin \
    --eval "db.runCommand({ping:1})" >/dev/null 2>&1 && break
  sleep 2
done

docker exec mongo mongosh -u admin -p "$ROOT_PW" --authenticationDatabase admin --eval "
db.getSiblingDB('market_analytics').createUser({
  user: 'writer',
  pwd: '$WRITER_PW',
  roles: [{ role: 'readWrite', db: 'market_analytics' }]
});" || true

# clusterMonitor + read on admin are required for mongo-express's own
# dashboard (it calls serverStatus/listCollections on admin on every page
# load) - without them every request hangs/500s even though the market_analytics
# read role alone is otherwise sufficient for browsing collections.
docker exec mongo mongosh -u admin -p "$ROOT_PW" --authenticationDatabase admin --eval "
db.getSiblingDB('market_analytics').createUser({
  user: 'public_reader',
  pwd: '$READER_PW',
  roles: [
    { role: 'read', db: 'market_analytics' },
    { role: 'clusterMonitor', db: 'admin' },
    { role: 'read', db: 'admin' }
  ]
});" || true
