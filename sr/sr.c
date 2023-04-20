#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "emulator.h"
#include "sr.h"

/* ******************************************************************
   Go Back N protocol.  Adapted from J.F.Kurose
   ALTERNATING BIT AND GO-BACK-N NETWORK EMULATOR: VERSION 1.2

   Network properties:
   - one way network delay averages five time units (longer if there
   are other messages in the channel for GBN), but can be larger
   - packets can be corrupted (either the header or the data portion)
   or lost, according to user-defined probabilities
   - packets will be delivered in the order in which they were sent
   (although some can be lost).

   Modifications:
   - removed bidirectional GBN code and other code not used by prac.
   - fixed C style to adhere to current programming style
   - added GBN implementation
**********************************************************************/

#define RTT  16.0       /* round trip time.  MUST BE SET TO 16.0 when submitting assignment */
#define WINDOWSIZE 6    /* the maximum number of buffered unacked packet */
#define SEQSPACE 12      /* the min sequence space for SR must be at least 2* windowsize */
#define NOTINUSE (-1)   /* used to fill header fields that are not being used */

/* generic procedure to compute the checksum of a packet.  Used by both sender and receiver
   the simulator will overwrite part of your packet with 'z's.  It will not overwrite your
   original checksum.  This procedure must generate a different checksum to the original if
   the packet is corrupted.
*/
int ComputeChecksum(struct pkt packet) {
    int checksum = 0;
    int i;

    checksum = packet.seqnum;
    checksum += packet.acknum;
    for (i = 0; i < 20; i++)
        checksum += (int) (packet.payload[i]);

    return checksum
}

bool IsCorrupted(struct pkt packet) {
    if (packet.checksum == ComputeChecksum(packet))
        return (false);
    else
        return (true);
}


/********* Sender (A) variables and functions ************/

static struct pkt buffer[WINDOWSIZE];  /* array for storing packets waiting for ACK */
static int windowfirst, windowlast;    /* array indexes of the first/last packet awaiting ACK */
static int windowcount;                /* the number of packets currently awaiting an ACK */
static int A_nextseqnum;               /* the next sequence number to be used by the sender */
static int ackcount;

/* called from layer 5 (application layer), passed the message to be sent to other side */
void A_output(struct msg message) {
    struct pkt sendpkt;
    int i;

    /* if not blocked waiting on ACK */
    if (windowcount + ackcount < WINDOWSIZE) {
        if (TRACE > 1)
            printf("----A: New message arrives, send window is not full, send new messge to layer3!\n");

        /* create packet */
        sendpkt.seqnum = A_nextseqnum;
        sendpkt.acknum = NOTINUSE;
        for (i = 0; i < 20; i++)
            sendpkt.payload[i] = message.data[i];
        sendpkt.checksum = ComputeChecksum(sendpkt);

        /* put packet in window buffer */
        /* windowlast will always be 0 for alternating bit; but not for GoBackN */
        windowlast = (windowlast + 1) % WINDOWSIZE;
        buffer[windowlast] = sendpkt;
        windowcount++;

        /* send out packet */
        if (TRACE > 0)
            printf("Sending packet %d to layer 3\n", sendpkt.seqnum);
        tolayer3(A, sendpkt);

        /* start timer if first packet in window */
        if (windowcount == 1)
            starttimer(A, RTT);

        /* get next sequence number, wrap back to 0 */
        A_nextseqnum = (A_nextseqnum + 1) % SEQSPACE;
    }
        /* if blocked,  window is full */
    else {
        if (TRACE > 0)
            printf("----A: New message arrives, send window is full\n");
        window_full++;
    }
}


/* called from layer 3, when a packet arrives for layer 4
   In this practical this will always be an ACK as B never sends data.
*/
void A_input(struct pkt packet) {

    int i;

    /* if received ACK is not corrupted */
    if (!IsCorrupted(packet)) {
        if (TRACE > 0)
            printf("----A: uncorrupted ACK %d is received\n", packet.acknum);
        total_ACKs_received++;

        /* check if new ACK or duplicate */
        if (windowcount != 0) {
            int seqfirst = buffer[windowfirst].seqnum;
            int seqlast = buffer[windowlast].seqnum;
            /* check case when seqnum has and hasn't wrapped */
            if (((seqfirst <= seqlast) && (packet.acknum >= seqfirst && packet.acknum <= seqlast)) ||
                ((seqfirst > seqlast) && (packet.acknum >= seqfirst || packet.acknum <= seqlast))) {

                /* packet is a new ACK */
                if (TRACE > 0)
                    printf("----A: ACK %d is not a duplicate\n", packet.acknum);
                new_ACKs++;

                /*check if new ACK or duplicate*/
                buffer[packet.acknum % WINDOWSIZE].acknum = 1;
                windowcount--;
                ackcount++;

                if (buffer[windowfirst].seqnum == packet.acknum) {
                    for (i = 0; i < WINDOWSIZE; i++) {
                        if (buffer[windowfirst].acknum == 1) {
                            windowfirst = (windowfirst + 1) % WINDOWSIZE;
                            ackcount--;
                        }
                    }
                    /* start timer again if there are still more unacked packets in window */
                    stoptimer(A);
                    if (windowcount > 0)
                        starttimer(A, RTT);
                }
            }
        } else if (TRACE > 0)
            printf("----A: duplicate ACK received, do nothing!\n");
    } else if (TRACE > 0)
        printf("----A: corrupted ACK is received, do nothing!\n");
}

/* called when A's timer goes off */
void A_timerinterrupt(void) {
    int i;
    if (TRACE > 0)
        printf("----A: time out,resend packets!\n");
    if (windowcount > 0) {
        for (i = 0; i < WINDOWSIZE; i++) {
            if (buffer[(windowfirst + i) % WINDOWSIZE].acknum != 1) {
                if (TRACE > 0)
                    printf("---A: resending packet %d\n", (buffer[(windowfirst + i) % WINDOWSIZE]).seqnum);

                tolayer3(A, buffer[(windowfirst + i) % WINDOWSIZE]);
                packets_resent++;

                starttimer(A, RTT);
                break;
            }
        }
    }
}



/* the following routine will be called once (only) before any other */
/* entity A routines are called. You can use it to do any initialization */
void A_init(void) {
    /* initialise A's window, buffer and sequence number */
    A_nextseqnum = 0;  /* A starts with seq num 0, do not change this */
    windowfirst = 0;
    windowlast = -1;   /* windowlast is where the last packet sent is stored.
		     new packets are placed in winlast + 1
		     so initially this is set to -1
		   */
    windowcount = 0;
    ackcount = 0;
}


/********* Receiver (B)  variables and procedures ************/

static int expectedseqnum; /* the sequence number expected next by the receiver */
static int B_nextseqnum;   /* the sequence number for the next packets sent by B */

static struct pkt B_buffer[WINDOWSIZE];
static int windowfirst_b;


/* called from layer 3, when a packet arrives for layer 4 at B*/
void B_input(struct pkt packet)
{

    struct pkt sendpkt;
    int i;

    int buffer_index=0;

    /* if not corrupted and received packet is in order */
    if ((!IsCorrupted(packet)))
    {
        if (TRACE > 0)
            printf("----B: packet %d is correctly received, send ACK!\n", packet.seqnum);
        sendpkt.acknum = packet.seqnum;
        sendpkt.seqnum = B_nextseqnum;
        B_nextseqnum = (B_nextseqnum + 1) % SEQSPACE;

        for (i = 0; i < 20; i++)
            sendpkt.payload[i] = '0';

        sendpkt.checksum = ComputeChecksum(sendpkt);

        tolayer3(B, sendpkt);

        packets_received++;

        buffer_index = (expectedseqnum - 1 + WINDOWSIZE) % SEQSPACE;

        if (((expectedseqnum <= buffer_index) && (packet.seqnum >= expectedseqnum && packet.seqnum <= buffer_index)) ||
            ((expectedseqnum > buffer_index) && (packet.acknum >= expectedseqnum || packet.acknum <= buffer_index))) {
            B_buffer[packet.seqnum % WINDOWSIZE] = packet;
            if (packet.seqnum == expectedseqnum) {
                for (i = 0; i < WINDOWSIZE; i++) {
                    if (B_buffer[windowfirst_b].seqnum == expectedseqnum) {
                        tolayer5(B, B_buffer[windowfirst].payload);
                        windowfirst_b = (windowfirst_b + 1) % WINDOWSIZE;
                        expectedseqnum = (expectedseqnum + 1) % SEQSPACE;
                    }
                }
            }
        }
    }
}

/* the following routine will be called once (only) before any other */
/* entity B routines are called. You can use it to do any initialization */
void B_init(void) {
    expectedseqnum = 0;
    B_nextseqnum = 1;
    windowfirst_b = 0;
}

/******************************************************************************
 * The following functions need be completed only for bi-directional messages *
 *****************************************************************************/

/* Note that with simplex transfer from a-to-B, there is no B_output() */
void B_output(struct msg message) {
}

/* called when B's timer goes off */
void B_timerinterrupt(void) {
}

