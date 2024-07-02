import json
from data_processing import extract_country_from_json

def get_all_wallet_attestations(cur_pg, attestations_csv):
    query = """
    SELECT "recipient", "schemaId", COUNT(*) as count, MAX("decodedDataJson") as decodedDataJson
    FROM "Attestation"
    WHERE "schemaId" IN (
        '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065',
        '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b',
        '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9',
        '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4'
    )
    GROUP BY "recipient", "schemaId";
    """
    cur_pg.execute(query)
    pg_attestations = cur_pg.fetchall()
    
    csv_attestations = attestations_csv[['recipient', 'schema.id', 'decodedDataJson']].values.tolist()
    
    return pg_attestations, csv_attestations

def create_user_profiles(cur_pg, attestations_csv):
    pg_attestations, csv_attestations = get_all_wallet_attestations(cur_pg, attestations_csv)
    profiles = {}

    # Process PostgreSQL attestations
    for recipient, schema_id, count, decoded_data_json in pg_attestations:
        if recipient not in profiles:
            profiles[recipient] = {
                "wallet": recipient,
                "country_code": "",
                "country": "",
                "activities": {"running": 0},
                "attended_events": [],
                "coinbase": False,
                "coinbase_one": False
            }
        
        profile = profiles[recipient]
        if schema_id == '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065':
            country_code, country_name = extract_country_from_json(decoded_data_json)
            profile["country_code"] = country_code
            profile["country"] = country_name
        elif schema_id == '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b':
            profile["activities"]["running"] += count
        elif schema_id == '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9':
            profile["coinbase"] = True
        elif schema_id == '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4':
            profile["coinbase_one"] = True

    # Process CSV attestations
    for recipient, schema_id, decoded_data_json in csv_attestations:
        if recipient not in profiles:
            profiles[recipient] = {
                "wallet": recipient,
                "country_code": "",
                "country": "",
                "activities": {"running": 0},
                "attended_events": [],
                "coinbase": False,
                "coinbase_one": False
            }
        
        profile = profiles[recipient]
        if schema_id == '0x1801901fabd0e6189356b4fb52bb0ab855276d84f7ec140839fbd1f6801ca065':
            country_code, country_name = extract_country_from_json(decoded_data_json)
            profile["country_code"] = country_code
            profile["country"] = country_name
        elif schema_id == '0x0f5b217904f3c65ad40b7af3db62716daddf53bb5db04b1a3ddb730fda0a474b':
            profile["activities"]["running"] += 1
        elif schema_id == '0xf8b05c79f090979bf4a80270aba232dff11a10d9ca55c4f88de95317970f0de9':
            profile["coinbase"] = True
        elif schema_id == '0x254bd1b63e0591fefa66818ca054c78627306f253f86be6023725a67ee6bf9f4':
            profile["coinbase_one"] = True

    users = list(profiles.values())

    print("User Profiles:")
    for user in users:
        print(user)

    return users
