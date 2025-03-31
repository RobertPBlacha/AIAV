#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

typedef long (*syscall_fn_t)(long, long, long, long, long, long, long);

static syscall_fn_t next_sys_call = NULL;

static pid_t tracked_pid = -1;						/* Store original soffice.bin PID */

/* Function to get the current timestamp in CSV format. */
void get_timestamp (char *buffer, size_t size) {
	struct timeval tv;
	gettimeofday(&tv, NULL);					/* Get current time in microseconds */

	struct tm *tm_info = localtime(&tv.tv_sec);			/* Convert to local time. */

	/* Format: YYYY-MM-DD HH:MM:SS.microseconds */
	strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
	snprintf(buffer + 19, size - 19,  ".%06ld", tv.tv_usec);	/* Append microseconds. */
	
}

/* Function to get the process name from /proc/<PID>/comm */
int is_target_process(pid_t pid) {

	if (tracked_pid == -1) {
		tracked_pid = pid;					/* Set the OG soffice.bin */
	}

	if (pid == tracked_pid) {
		return 1;
	}

	char path[64];
	snprintf(path, sizeof(path), "/proc/%d/status", pid);

	FILE *file = fopen(path, "r");
	if (!file) return 0;						/* If we can't read the file, return false. */

	char line[128];
	while (fgets(line, sizeof(line), file)) {
		if (strncmp(line, "PPid:", 5) == 0) {
			int ppid;
			sscanf(line, "PPid:\t%d", &ppid);
			fclose(file);
			return (ppid == tracked_pid);			/* Return true if process is a child */
		}
	}
	fclose(file);
	/*return strcmp(proc_name, target) == 0;*/				/* Return true if matches */
	return 0;
}


static long hook_function(long a1, long a2, long a3,
			  long a4, long a5, long a6,
			  long a7)
{
	pid_t pid = getpid();
	if (!is_target_process(pid)) {
		return next_sys_call(a1, a2, a3, a4, a5, a6, a7);	/* Skip logging. */
	}

	char timestamp[32];
	get_timestamp(timestamp, sizeof(timestamp));			/* Get formatted timestamp */

	FILE *logfile = fopen("/tmp/zpoline_syscalls.csv", "a");
	if (logfile) {
		fprintf(logfile, "%s,%ld,%d\n", timestamp, a1, getpid());
		fflush(logfile);
		fclose(logfile);
	}

	return next_sys_call(a1, a2, a3, a4, a5, a6, a7);
}

int __hook_init(long placeholder __attribute__((unused)),
		void *sys_call_hook_ptr)
{
	

	FILE *logfile = fopen("/tmp/zpoline_syscalls.csv", "w");
	if (logfile) {
		fprintf(logfile, "TIMESTAMP,SYSCALL_NUMBER,PID\n");
		fflush(logfile);
		fclose(logfile);
	}

	next_sys_call = *((syscall_fn_t *) sys_call_hook_ptr);
	*((syscall_fn_t *) sys_call_hook_ptr) = hook_function;

	return 0;
}
