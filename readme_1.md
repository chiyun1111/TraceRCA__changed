##  File List

`API.tar.gz`, `container.tar.gz` and `microservice.tar.gz` contain all injected faults.

`noinjection` contains some data without fault injection.



## File Format

All files are in Python pickle format, which means they should be load with `pickle.load(file)`.

In each pickle file, there is a list of objects. Each object represents a trace, and here is a example:

``` python
[{'trace_id': '000174280f4e7c489078deeb005471c2',
 'timestamp': [1570509359757708.0],
 'latency': [855.0],
 'http_status': [304.0],
 'cpu_use': [0.07737251774500982],
 'mem_use_percent': [0.008921875],
 'mem_use_amount': [120233984.0],
 'file_write_rate': [0.0],
 'file_read_rate': [0.0],
 'net_send_rate': [7036.555073177282],
 'net_receive_rate': [3164.302018529293],
 'endtime': [1570509359758563.0],
 's_t': [('ts-ui-dashboard', 'ts-ui-dashboard')],
 'label': 0}]
```

