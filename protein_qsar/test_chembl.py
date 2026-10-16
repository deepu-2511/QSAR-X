from chembl_webresource_client.new_client import new_client

target = new_client.target

results = target.search("EGFR")

print("\nChEMBL target search results:\n")

for item in results[:10]:
    print(
        item.get("target_chembl_id"),
        "|",
        item.get("pref_name"),
        "|",
        item.get("target_type"),
        "|",
        item.get("organism")
    )