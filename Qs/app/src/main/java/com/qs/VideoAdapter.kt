package com.qs

import android.content.Context
import android.net.Uri
import android.util.Log
import android.util.SparseArray
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import androidx.recyclerview.widget.RecyclerView

class VideoAdapter(
    private val context: Context,
    private val videos: List<Uri>,
    private val historyStore: HistoryStore
) : RecyclerView.Adapter<VideoAdapter.VH>() {

    private val players = SparseArray<ExoPlayer>()
    private var currentPos = 0

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val playerView: PlayerView = view.findViewById(R.id.playerView)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_video, parent, false)
        return VH(view)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val uri = videos[position]
        var player = players.get(position)
        
        if (player == null) {
            player = ExoPlayer.Builder(context).build()
            player.setMediaItem(MediaItem.fromUri(uri))
            player.repeatMode = Player.REPEAT_MODE_ONE
            player.prepare()
            
            val saved = historyStore.getPosition(uri.toString())
            if (saved > 0) player.seekTo(saved)
            
            // 加上错误监听，如果解码失败会直接弹提示
            player.addListener(object : Player.Listener {
                override fun onPlayerError(error: PlaybackException) {
                    super.onPlayerError(error)
                    Log.e("QsPlayer", "播放失败: ${error.errorCodeName}", error)
                    Toast.makeText(context, "播放失败: ${error.errorCodeName}", Toast.LENGTH_LONG).show()
                }
            })
            
            players.put(position, player)
        }
        
        holder.playerView.player = player
        if (position == currentPos) {
            player.play()
        } else {
            player.pause()
        }

        holder.itemView.setOnClickListener {
            if (player.isPlaying) player.pause() else player.play()
        }
    }

    override fun onViewRecycled(holder: VH) {
        super.onViewRecycled(holder)
        val pos = holder.bindingAdapterPosition
        if (pos != RecyclerView.NO_POSITION) {
            players.get(pos)?.let {
                historyStore.savePosition(videos[pos].toString(), it.currentPosition)
                it.release()
                players.remove(pos)
            }
        }
        holder.playerView.player = null
    }

    override fun getItemCount() = videos.size

    fun setCurrent(position: Int) {
        players.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
            it.pause()
        }
        currentPos = position
        players.get(position)?.play()
    }

    fun saveCurrentPosition() {
        players.get(currentPos)?.let {
            historyStore.savePosition(videos[currentPos].toString(), it.currentPosition)
        }
    }

    fun releaseAll() {
        for (i in 0 until players.size()) {
            val p = players.valueAt(i)
            historyStore.savePosition(videos[players.keyAt(i)].toString(), p.currentPosition)
            p.release()
        }
        players.clear()
    }
}
