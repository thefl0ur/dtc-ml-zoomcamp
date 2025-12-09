build:
	docker build -t hw9 .

run:
	docker run -d -p 8080:8080 hw9

get:
	curl -X POST "http://localhost:8080/2015-03-31/functions/function/invocations" \
	 -d '{"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}'