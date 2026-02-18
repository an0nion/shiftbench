"""List all datasets in the ShiftBench registry."""

from shiftbench.data import DatasetRegistry

def main():
    registry = DatasetRegistry()

    # Get all datasets
    all_datasets = registry.list_datasets()

    print(f"ShiftBench Dataset Registry")
    print("=" * 80)
    print(f"\nTotal datasets: {len(all_datasets)}")

    # List by domain
    print("\n" + "=" * 80)
    print("Datasets by Domain")
    print("=" * 80)

    for domain in sorted(registry.get_domains()):
        domain_datasets = registry.list_datasets(domain=domain)
        print(f"\n{domain.upper()} ({len(domain_datasets)} datasets):")
        for name in sorted(domain_datasets):
            info = registry.get_dataset_info(name)
            print(f"  - {name:20s} | {info['n_samples']:>6d} samples | "
                  f"{info['n_features']:>5d} features | {info['shift_type']:20s}")

    # Get metadata
    metadata = registry.get_metadata()
    print("\n" + "=" * 80)
    print("Registry Metadata")
    print("=" * 80)
    print(f"Status: {metadata.get('status', 'Unknown')}")

    domains = metadata.get('domains', {})
    print(f"\nDatasets by domain:")
    for domain, count in domains.items():
        print(f"  - {domain}: {count}")

    expansion_plan = metadata.get('expansion_plan', {})
    if expansion_plan:
        print(f"\nExpansion plan:")
        for key, value in expansion_plan.items():
            print(f"  - {key}: {value}")

    recent_updates = metadata.get('recent_updates', [])
    if recent_updates:
        print(f"\nRecent updates:")
        for update in recent_updates:
            print(f"  - {update.get('date', 'Unknown')}: {update.get('update', 'No description')}")

if __name__ == "__main__":
    main()
