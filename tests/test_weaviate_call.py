import weaviate
from weaviate.classes.config import Property, DataType, Configure

client = weaviate.connect_to_local(host="localhost", port=22006, grpc_port=50051)

name = "TestNoVec"
try:
    client.collections.get(name)
except:
    client.collections.create(
        name=name,
        properties=[Property(name="text", data_type=DataType.TEXT)],
        vectorizer_config=Configure.Vectorizer.none(),
    )

col = client.collections.get(name)

# 插入一条（不需要 embedding）
col.data.insert({"text": "hello weaviate"})
print("insert ok")

print("count:", col.aggregate.over_all(total_count=True).total_count)
client.close()
