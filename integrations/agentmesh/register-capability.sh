#!/bin/sh
set -eu
: "${MEDAI_AGENTMESH_CATALOG:=${AGENTMESH_CAPABILITY_CATALOG:?Set MEDAI_AGENTMESH_CATALOG to the AgentMesh capability catalog}}"
command -v agentmesh >/dev/null 2>&1 || { echo "agentmesh CLI is required" >&2; exit 127; }
for package in integrations/agentmesh/capabilities/*.yaml; do
  agentmesh capabilities validate "$package"
  agentmesh capabilities install "$package" --catalog "$MEDAI_AGENTMESH_CATALOG"
done
echo "Registered MedAI capabilities in $MEDAI_AGENTMESH_CATALOG"
