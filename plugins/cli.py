"""
CLI interface for the Scan2Measure Plugin Marketplace.

Usage:
    python -m plugins.cli browse [--type TYPE]
    python -m plugins.cli search QUERY
    python -m plugins.cli info NAME
    python -m plugins.cli add NAME
    python -m plugins.cli remove NAME
    python -m plugins.cli list
    python -m plugins.cli enable NAME
    python -m plugins.cli disable NAME
    python -m plugins.cli categories
"""

import argparse
import sys

from plugins.marketplace import PluginMarketplace


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scan2measure-plugins",
        description="Scan2Measure Plugin Marketplace",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # browse
    p_browse = sub.add_parser("browse", help="Browse available plugins")
    p_browse.add_argument("--type", "-t", dest="plugin_type", default=None,
                          help="Filter by plugin type (e.g. feature_extraction)")

    # search
    p_search = sub.add_parser("search", help="Search plugins by keyword")
    p_search.add_argument("query", help="Search term")

    # info
    p_info = sub.add_parser("info", help="Show detailed plugin information")
    p_info.add_argument("name", help="Plugin name")

    # add (install)
    p_add = sub.add_parser("add", help="Install a plugin from the marketplace")
    p_add.add_argument("name", help="Plugin name to install")

    # remove (uninstall)
    p_remove = sub.add_parser("remove", help="Uninstall a plugin")
    p_remove.add_argument("name", help="Plugin name to uninstall")

    # list
    sub.add_parser("list", help="List installed plugins")

    # enable
    p_enable = sub.add_parser("enable", help="Enable a disabled plugin")
    p_enable.add_argument("name", help="Plugin name")

    # disable
    p_disable = sub.add_parser("disable", help="Disable a plugin without uninstalling")
    p_disable.add_argument("name", help="Plugin name")

    # categories
    sub.add_parser("categories", help="List all plugin categories")

    args = parser.parse_args(argv)
    marketplace = PluginMarketplace()

    if args.command == "browse":
        marketplace.print_catalog(args.plugin_type)

    elif args.command == "search":
        results = marketplace.search(args.query)
        if results:
            print(f"\n  Search results for '{args.query}':")
            print("  " + "-" * 50)
            for p in results:
                installed = marketplace.registry.is_installed(p["name"])
                tag = " [installed]" if installed else ""
                print(f"  {p['name']} v{p['version']}{tag}")
                print(f"    {p['description']}")
            print()
        else:
            print(f"  No plugins found matching '{args.query}'.")

    elif args.command == "info":
        info = marketplace.get_info(args.name)
        if info:
            print(f"\n  Plugin: {info['name']}")
            print("  " + "=" * 50)
            print(f"  Version:      {info['version']}")
            print(f"  Description:  {info['description']}")
            print(f"  Author:       {info['author']}")
            print(f"  Type:         {info['plugin_type']}")
            print(f"  License:      {info.get('license', 'N/A')}")
            print(f"  Dependencies: {', '.join(info.get('dependencies', [])) or 'none'}")
            print(f"  Tags:         {', '.join(info.get('tags', [])) or 'none'}")
            print(f"  Installed:    {'yes' if info.get('installed') else 'no'}")
            print()
        else:
            print(f"  Plugin '{args.name}' not found in marketplace.")

    elif args.command == "add":
        marketplace.install(args.name)

    elif args.command == "remove":
        marketplace.uninstall(args.name)

    elif args.command == "list":
        marketplace.print_installed()

    elif args.command == "enable":
        if marketplace.registry.enable(args.name):
            print(f"  Enabled '{args.name}'.")
        else:
            print(f"  Plugin '{args.name}' not found.")

    elif args.command == "disable":
        if marketplace.registry.disable(args.name):
            print(f"  Disabled '{args.name}'.")
        else:
            print(f"  Plugin '{args.name}' not found.")

    elif args.command == "categories":
        cats = marketplace.get_categories()
        print("\n  Plugin Categories:")
        print("  " + "-" * 30)
        for c in cats:
            count = len(marketplace.browse(c))
            print(f"  {c:<25} ({count} available)")
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
