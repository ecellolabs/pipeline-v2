#!/usr/bin/env bash
set -euo pipefail

# Usage: ./bump.sh [major|minor|patch]
BUMP_TYPE=${1:-patch}  # default to patch if no argument provided

# 1️⃣ Make sure the working directory is clean
if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ Working directory is not clean. Commit or stash your changes first."
  exit 1
fi

# 2️⃣ Make sure we are on main
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
  echo "❌ You must be on main branch. Current: $CURRENT_BRANCH"
  exit 1
fi

# 3️⃣ Bump version with uv
echo "🔧 Bumping version ($BUMP_TYPE)..."
NEW_VERSION=$(uv version --bump "$BUMP_TYPE" --dry-run | awk '{print $2}')
uv version --bump "$BUMP_TYPE"
echo "✅ New version: $NEW_VERSION"

# 4️⃣ Commit the bump
echo "💾 Committing version bump..."
git add pyproject.toml
git commit -m "chore: bump version to $NEW_VERSION"

# 5️⃣ Create Git tag
TAG="v$NEW_VERSION"
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "⚠️ Tag $TAG already exists, skipping tag creation."
else
  echo "🏷️ Creating Git tag $TAG..."
  git tag "$TAG"
fi

# 6️⃣ Push commit and tag
echo "🚀 Pushing commit and tag..."
git push origin "$TAG"

echo "🎉 Done! Version bumped to $NEW_VERSION and tag pushed."
